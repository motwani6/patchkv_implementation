# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short Colab-only checkpoint/eviction/restore smoke test; no benchmarks."""

import argparse
import json
import os
import traceback
from pathlib import Path
from uuid import uuid4

from suffix_repair_test.patchkv_colab import command, save


def run(args, report, persist):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5):
        raise RuntimeError(
            "Use a supported Colab CUDA GPU (compute capability >= 7.5)."
        )
    report.update(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        nvidia_smi=command("nvidia-smi"),
    )
    from vllm import LLM, SamplingParams
    from vllm.v1.core.patchkv import (
        plan_attention_guided_suffix_repair,
        plan_profiled_suffix_repair,
        plan_suffix_repair,
    )
    from vllm.v1.core.patchkv_attention import AttentionBudget
    from vllm.v1.core.patchkv_checkpoint import CPUCheckpointStore
    from vllm.v1.core.patchkv_profile import DriftProfile

    config = dict(
        model=args.model,
        revision=args.revision,
        dtype="float16",
        enforce_eager=True,
        max_model_len=192,
        max_num_seqs=1,
        max_num_batched_tokens=32,
        block_size=16,
        enable_prefix_caching=True,
        kv_cache_memory_bytes=32 << 20,
        gpu_memory_utilization=0.6,
        attention_config={"backend": "TRITON_ATTN"},
    )
    report["requested_config"] = config
    profile = DriftProfile.load(args.drift_profile) if args.drift_profile else None
    model_key = f"{args.model}@{args.revision};fp16;TRITON_ATTN;maxlen=192"
    if profile is not None and (
        profile.config.model_key != model_key or profile.config.block_size != 16
    ):
        raise ValueError("Drift profile does not match this smoke configuration.")
    if profile is not None:
        report["drift_profile"] = json.loads(
            args.drift_profile.read_text(encoding="utf-8")
        )
    report["attention_fraction"] = args.attention_fraction
    persist()
    llm = LLM(**config)
    try:
        core = llm.llm_engine.engine_core.engine_core
        worker = core.model_executor.driver_worker
        manager = core.scheduler.kv_cache_manager
        editor = manager.editor
        if editor is None:
            raise RuntimeError("Unsupported checkpoint/editor configuration.")
        context = worker.vllm_config.compilation_config.static_forward_context
        model = worker.get_model()
        report["effective_config"] = str(worker.vllm_config)
        tokenizer = llm.get_tokenizer()
        original = tokenizer.encode(
            "Remember the blue birds in the quiet garden. " * 16,
            add_special_tokens=False,
        )[:83]
        if len(original) != 83:
            raise ValueError("Expected an 83-token test prompt.")
        params = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)

        def snapshot(ids, salt):
            request = editor._request(ids, salt)
            pages = []
            for lo in range(0, len(ids), 16):
                hit = editor.pool.get_cached_block(
                    editor._hash_at(request, min(lo + 16, len(ids))), [0]
                )
                if hit is None:
                    raise AssertionError("Expected published KV page is missing.")
                pages.append(hit[0].block_id)
            return {
                name: torch.cat(
                    [context[name].kv_cache[page].transpose(0, 1) for page in pages]
                )[: len(ids)]
                .detach()
                .cpu()
                .clone()
                for name in manager.kv_cache_config.kv_cache_groups[0].layer_names
            }

        def fresh(ids, salt):
            output = llm.generate(
                {"prompt_token_ids": ids, "cache_salt": salt}, params, use_tqdm=False
            )[0]
            if output.num_cached_tokens:
                raise AssertionError("Fresh prefill unexpectedly reused KV.")
            return snapshot(ids, salt)

        for label, start, end, text in (
            ("insertion", 19, 19, " red green"),
            ("deletion", 19, 23, ""),
            ("replacement", 19, 21, " red green"),
        ):
            replacement = (
                tokenizer.encode(text, add_special_tokens=False) if text else []
            )
            edited_ids = original[:start] + replacement + original[end:]
            row = dict(
                case=label,
                status="running",
                modes=[],
                original_ids=original,
                replacement_ids=replacement,
                edited_ids=edited_ids,
                start=start,
                end=end,
            )
            report["cases"].append(row)
            persist()
            reference = fresh(edited_ids, uuid4().hex)
            source_salt = uuid4().hex
            attention = None
            if args.attention_fraction is not None:
                from vllm.v1.worker.patchkv_attention import (
                    CheckpointAttentionCollector,
                )

                names = manager.kv_cache_config.kv_cache_groups[0].layer_names
                capture = CheckpointAttentionCollector(
                    worker.model_runner,
                    original,
                    model_key=model_key,
                    checkpoint_id=source_salt,
                    block_size=16,
                    layer_names=[
                        names[i] for i in sorted({0, len(names) // 2, len(names) - 1})
                    ],
                    query_positions=[len(original) - 1],
                )
                with capture:
                    before = fresh(original, source_salt)
                attention = capture.finalize()
            else:
                before = fresh(original, source_salt)
            checkpoint = llm.create_kv_checkpoint(
                original,
                cache_salt=source_salt,
                model_key=model_key,
                max_cpu_bytes=16 << 20,
                attention=attention,
            )
            source_request = editor._request(original, source_salt)
            source_page_ids = [
                editor.pool.get_cached_block(
                    editor._hash_at(source_request, min(lo + 16, len(original))), [0]
                )[0].block_id
                for lo in range(0, len(original), 16)
            ]
            try:
                checkpoint.metadata.save(args.results_dir / f"checkpoint-{label}.json")
                row["checkpoint_id"] = checkpoint.checkpoint_id
                row["payload_bytes"] = checkpoint.metadata.payload_bytes
                row["charged_cpu_bytes"] = CPUCheckpointStore.charged_bytes(
                    checkpoint.metadata
                )
                suffix_start = start + len(replacement)
                blocks = sorted({i // 16 for i in range(suffix_start, len(edited_ids))})
                modes = [
                    ("direct", {}),
                    ("selective", dict(repair_suffix_blocks=[blocks[0], blocks[-1]])),
                    ("full", dict(recompute_suffix=True)),
                ]
                if profile is not None:
                    modes.append(("predicted", dict(drift_profile=profile)))
                if attention is not None:
                    modes.append(
                        (
                            "attention_guided",
                            dict(
                                drift_profile=profile,
                                attention_budget=AttentionBudget(
                                    fraction=args.attention_fraction
                                ),
                            ),
                        )
                    )
                for mode, options in modes:
                    item = {"mode": mode, "status": "running"}
                    row["modes"].append(item)
                    persist()
                    if not llm.reset_prefix_cache():
                        raise RuntimeError("Could not evict all prefix-cache aliases.")
                    # Quiescent test only: destroy all original source pages.
                    # Leave the backend's null/padding page untouched.
                    for layer in context.values():
                        if hasattr(layer, "kv_cache"):
                            indices = torch.tensor(
                                source_page_ids, device=layer.kv_cache.device
                            )
                            layer.kv_cache.index_fill_(0, indices, float("nan"))
                    worker.synchronize_device()
                    request = editor._request(original, source_salt)
                    if any(
                        editor.pool.get_cached_block(
                            editor._hash_at(request, min(lo + 16, len(original))), [0]
                        )
                        is not None
                        for lo in range(0, len(original), 16)
                    ):
                        raise AssertionError("Source aliases survived eviction.")
                    item["source_aliases_evicted_and_source_pages_destroyed"] = True
                    forwards = []

                    def record(module, inputs, kwargs, forwards=forwards):
                        forwards.extend(kwargs["positions"].tolist())

                    with model.register_forward_pre_hook(record, with_kwargs=True):
                        prompt = llm.edit_kv_cache(
                            original, text, start, end, checkpoint=checkpoint, **options
                        )
                    if mode == "attention_guided":
                        plan = plan_attention_guided_suffix_repair(
                            original,
                            start,
                            end,
                            len(replacement),
                            16,
                            profile,
                            attention,
                            options["attention_budget"],
                            model_key=model_key,
                            checkpoint_id=source_salt,
                        )
                    elif mode == "predicted":
                        plan = plan_profiled_suffix_repair(
                            len(original),
                            start,
                            end,
                            len(replacement),
                            16,
                            profile,
                            model_key=model_key,
                        )
                    else:
                        plan = plan_suffix_repair(
                            len(original),
                            start,
                            end,
                            len(replacement),
                            16,
                            blocks
                            if mode == "full"
                            else options.get("repair_suffix_blocks", []),
                        )
                    expected_positions = list(range(start, suffix_start)) + [
                        i for lo, hi in plan.repair_ranges for i in range(lo, hi)
                    ]
                    if forwards != expected_positions:
                        raise AssertionError(
                            "Unexpected checkpoint repair forward positions."
                        )
                    item["forward_positions"] = forwards
                    actual = snapshot(edited_ids, prompt["cache_salt"])
                    maximum = 0.0
                    for name, kv in actual.items():
                        torch.testing.assert_close(
                            kv[:start], before[name][:start], atol=0, rtol=0
                        )
                        if not plan.reuse_blocks:
                            maximum = max(
                                maximum, (kv - reference[name]).abs().max().item()
                            )
                            torch.testing.assert_close(
                                kv, reference[name], atol=0.03, rtol=0.03
                            )
                        reused = [
                            i
                            for i in range(suffix_start, len(edited_ids))
                            if i // 16 in plan.reuse_blocks
                        ]
                        if reused:
                            rope = model.get_submodule(
                                name.rsplit(".", 1)[0]
                            ).rotary_emb
                            if (
                                not rope.is_neox_style
                                or rope.rotary_dim != rope.head_size
                            ):
                                raise RuntimeError(
                                    "Smoke oracle requires full NeoX RoPE."
                                )
                            old = torch.tensor(reused) - (
                                len(replacement) - (end - start)
                            )
                            new = torch.tensor(reused)
                            head = rope.head_size
                            source = before[name][old]
                            torch.testing.assert_close(
                                kv[new, :, head:], source[..., head:], atol=0, rtol=0
                            )
                            phase = torch.complex(
                                *rope.cos_sin_cache.float().cpu().chunk(2, -1)
                            )
                            key = torch.complex(
                                *source[..., :head].float().chunk(2, -1)
                            )
                            moved = key * (phase[new] / phase[old])[:, None, :]
                            expected = torch.cat((moved.real, moved.imag), -1)
                            torch.testing.assert_close(
                                kv[new, :, :head].float(),
                                expected,
                                atol=0.005,
                                rtol=0.002,
                            )
                    item["max_abs_error_vs_fresh"] = (
                        maximum if not plan.reuse_blocks else None
                    )
                    output = llm.generate(
                        prompt,
                        SamplingParams(max_tokens=2, temperature=0, ignore_eos=True),
                        use_tqdm=False,
                    )[0]
                    if (
                        len(output.outputs[0].token_ids) != 2
                        or output.num_cached_tokens != len(edited_ids) - 1
                    ):
                        raise AssertionError(
                            "Restored cache was not reused for generation."
                        )
                    item.update(
                        status="passed",
                        generated_ids=output.outputs[0].token_ids,
                        generated_text=output.outputs[0].text,
                        cached_tokens=output.num_cached_tokens,
                    )
                    persist()
            finally:
                checkpoint.release()
            if llm._patchkv_checkpoint_session.store.used_bytes != 0:
                raise AssertionError(
                    "Released checkpoint still owns accounted CPU memory."
                )
            row["status"] = "passed"
            persist()
    finally:
        llm.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--revision", default="c89bee90d9f811437d9735454613c35b4a3c4dc8"
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--drift-profile", type=Path)
    parser.add_argument("--attention-fraction", type=float)
    args = parser.parse_args()
    if args.attention_fraction is not None and (
        not args.drift_profile or not 0 <= args.attention_fraction <= 1
    ):
        parser.error(
            "Attention recovery needs a drift profile and a fraction in [0, 1]."
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    args.results_dir.mkdir(parents=True, exist_ok=False)
    path = args.results_dir / "checkpoint-results.json"
    root = Path(__file__).resolve().parents[1]
    report = dict(
        status="running",
        cases=[],
        git_head=command("git", "rev-parse", "HEAD"),
        git_status=command("git", "status", "--short"),
        source_sha256={
            str(p.relative_to(root)): hashlib_sha(p)
            for p in (
                root / "vllm/entrypoints/patchkv.py",
                root / "vllm/entrypoints/llm.py",
                root / "vllm/v1/core/patchkv_checkpoint.py",
                root / "vllm/v1/worker/patchkv_checkpoint.py",
                Path(__file__),
            )
        },
    )
    persist = lambda: save(path, report)
    persist()
    try:
        run(args, report, persist)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        persist()
        print(f"Results: {path} ({report['status']})", flush=True)


def hashlib_sha(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
