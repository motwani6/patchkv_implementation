# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small, fail-fast GPU validation; see COLAB_PATCHKV.md for setup."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as error:
        return {"returncode": None, "stdout": "", "stderr": str(error)}
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def save(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate(args, report, persist):
    import torch

    report["torch"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select a Colab GPU runtime.")
    capability = torch.cuda.get_device_capability()
    report["gpu"] = torch.cuda.get_device_name()
    report["capability"] = capability
    report["compiled_arches"] = torch.cuda.get_arch_list()
    if capability < (7, 5):
        raise RuntimeError("This checkout requires compute capability >= 7.5.")
    # Exercise allocation, a CUDA library, and synchronization before model load.
    x = torch.ones((16, 16), device="cuda", dtype=torch.float16)
    if (x @ x)[0, 0].item() != 16:
        raise RuntimeError("CUDA matmul smoke check failed.")
    torch.cuda.synchronize()
    import vllm._C_stable_libtorch  # noqa: F401

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.v1.core.patchkv import (
        plan_attention_guided_suffix_repair,
        plan_profiled_suffix_repair,
        plan_suffix_repair,
    )
    from vllm.v1.core.patchkv_attention import AttentionBudget, AttentionMetadata
    from vllm.v1.core.patchkv_profile import DriftProfile

    root = Path(__file__).resolve().parents[1]
    report["vllm"] = {"version": vllm.__version__, "path": vllm.__file__}
    if Path(vllm.__file__).resolve().parent != root / "vllm":
        raise RuntimeError("Imported vLLM is not this modified checkout.")
    persist()
    if args.preflight_only:
        return
    drift_profile = (
        DriftProfile.load(args.drift_profile) if args.drift_profile else None
    )
    model_key = f"{args.model}@{args.revision};fp16;TRITON_ATTN;maxlen=192"
    if drift_profile is not None and (
        drift_profile.config.model_key != model_key
        or drift_profile.config.block_size != 16
    ):
        raise ValueError("Drift profile does not match this smoke-test configuration.")
    if drift_profile is not None:
        report["drift_profile"] = json.loads(
            args.drift_profile.read_text(encoding="utf-8")
        )
    config = dict(
        model=args.model,
        revision=args.revision,
        enforce_eager=True,
        dtype="float16",
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
    persist()
    llm = LLM(**config)
    try:
        core = llm.llm_engine.engine_core.engine_core
        manager = core.scheduler.kv_cache_manager
        editor = manager.editor
        if editor is None:
            raise RuntimeError("KV editor disabled by effective configuration.")
        worker = core.model_executor.driver_worker
        model = worker.get_model()
        context = worker.vllm_config.compilation_config.static_forward_context
        report["effective_config"] = str(worker.vllm_config)
        report["runner"] = type(worker.model_runner).__name__
        report["model_commit"] = getattr(
            worker.vllm_config.model_config.hf_config, "_commit_hash", None
        )
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True)
        warmup = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
        tokenizer = llm.get_tokenizer()
        original = tokenizer.encode(
            "Remember the color blue. Count the small birds in the garden. " * 12,
            add_special_tokens=False,
        )[:83]
        if len(original) != 83:
            raise RuntimeError("Expected an 83-token test context.")

        def generate(prompt, sampling=warmup):
            return llm.generate(prompt, sampling, use_tqdm=False)[0]

        def snapshot(ids, salt):
            request = editor._request(ids, salt)
            pages = []
            for pos in range(0, len(ids), editor.block_size):
                end = min(pos + editor.block_size, len(ids))
                hit = editor.pool.get_cached_block(editor._hash_at(request, end), [0])
                if hit is None:
                    raise RuntimeError(f"Source evicted or unpublished at {end}.")
                pages.append(hit[0])
            states = {}
            for name in manager.kv_cache_config.kv_cache_groups[0].layer_names:
                cache = context[name].kv_cache
                if cache.ndim != 4:
                    raise RuntimeError(
                        f"Unsupported cache layout: {name}: {cache.shape}"
                    )
                states[name] = torch.cat(
                    [cache[b.block_id].transpose(0, 1) for b in pages]
                )[: len(ids)].clone()
            return states, pages

        def compare(actual, expected, exact=False):
            maximum = 0.0
            for name in expected:
                maximum = max(
                    maximum, (actual[name] - expected[name]).abs().max().item()
                )
                torch.testing.assert_close(
                    actual[name],
                    expected[name],
                    atol=0 if exact else args.atol,
                    rtol=0 if exact else args.rtol,
                    msg=lambda msg, name=name: f"{name}: {msg}",
                )
            return maximum

        for label, start, end, text in (
            ("replacement", 19, 23, " red green"),
            ("insertion", 19, 19, " red green"),
            ("deletion", 19, 23, ""),
        ):
            row = {"case": label, "status": "running", "modes": []}
            report["cases"].append(row)
            replacement = tokenizer.encode(text) if text else []
            if label == "replacement":
                end = start + len(replacement)
            edited_ids = original[:start] + replacement + original[end:]
            suffix_start = start + len(replacement)
            blocks = sorted({p // 16 for p in range(suffix_start, len(edited_ids))})
            selected = [blocks[0], blocks[-1]]
            row.update(
                original_ids=original,
                replacement_ids=replacement,
                edited_ids=edited_ids,
                start=start,
                end=end,
                shift=len(replacement) - (end - start),
                selected=selected,
            )
            persist()
            # Unique salts ensure the reference starts with no cached context.
            fresh_salt, source_salt = uuid4().hex, uuid4().hex
            fresh_output = generate(
                {"prompt_token_ids": edited_ids, "cache_salt": fresh_salt}
            )
            if fresh_output.num_cached_tokens != 0:
                raise AssertionError("Fresh reference unexpectedly hit the cache.")
            fresh, _ = snapshot(edited_ids, fresh_salt)
            attention = None
            if args.attention_fraction is None:
                generate({"prompt_token_ids": original, "cache_salt": source_salt})
            else:
                from vllm.v1.worker.patchkv_attention import (
                    CheckpointAttentionCollector,
                )

                names = manager.kv_cache_config.kv_cache_groups[0].layer_names
                sampled = args.attention_layers or [
                    names[i] for i in sorted({0, len(names) // 2, len(names) - 1})
                ]
                query_positions = args.attention_query_positions or [len(original) - 1]
                if max(query_positions) >= 190:
                    raise ValueError("Attention probes must fit the 192-token context.")
                capture = CheckpointAttentionCollector(
                    worker.model_runner,
                    original,
                    model_key=model_key,
                    checkpoint_id=source_salt,
                    block_size=16,
                    layer_names=sampled,
                    query_positions=query_positions,
                )
                probe_tokens = max(1, max(query_positions) - len(original) + 2)
                with capture:
                    probe_output = generate(
                        {"prompt_token_ids": original, "cache_salt": source_salt},
                        SamplingParams(
                            max_tokens=probe_tokens, temperature=0, ignore_eos=True
                        ),
                    )
                captured = capture.finalize()
                attention_path = args.results_dir / f"attention-{label}.json"
                captured.save(attention_path)
                attention = AttentionMetadata.load(attention_path)
                attention.validate_context(
                    original,
                    model_key=model_key,
                    block_size=16,
                    checkpoint_id=source_salt,
                    layers=captured.layers,
                    query_positions=captured.query_positions,
                )
                row["attention_metadata"] = attention_path.name
                row["attention_probe_generated_ids"] = probe_output.outputs[0].token_ids
                persist()
            before, source = snapshot(original, source_salt)
            editor.pool.touch(source)
            try:
                modes = [
                    ("full", {"recompute_suffix": True}),
                    ("all_blocks", {"repair_suffix_blocks": blocks}),
                    ("selective", {"repair_suffix_blocks": selected}),
                    ("direct", {}),
                ]
                if drift_profile is not None:
                    predicted = plan_profiled_suffix_repair(
                        len(original),
                        start,
                        end,
                        len(replacement),
                        16,
                        drift_profile,
                        model_key=model_key,
                    )
                    modes.append(
                        ("predicted", {"repair_suffix_blocks": predicted.repair_blocks})
                    )
                    row["predicted_dirty_tokens"] = drift_profile.predict(
                        start, len(replacement), len(original) - end
                    )
                    if attention is not None:
                        budget = AttentionBudget(fraction=args.attention_fraction)
                        guided = plan_attention_guided_suffix_repair(
                            original,
                            start,
                            end,
                            len(replacement),
                            16,
                            drift_profile,
                            attention,
                            budget,
                            model_key=model_key,
                            checkpoint_id=source_salt,
                        )
                        modes.append(
                            (
                                "attention_guided",
                                {"repair_suffix_blocks": guided.repair_blocks},
                            )
                        )
                        row["attention_selected_suffix_offsets"] = (
                            attention.select_suffix_offsets(
                                end,
                                row["predicted_dirty_tokens"],
                                budget,
                            )
                        )
                        row["attention_budget_fraction"] = args.attention_fraction
                for mode, options in modes:
                    item = {"mode": mode, "status": "running"}
                    row["modes"].append(item)
                    persist()
                    forwards = []

                    def record(module, inputs, kwargs, forwards=forwards):
                        forwards.extend(kwargs["positions"].tolist())

                    with model.register_forward_pre_hook(record, with_kwargs=True):
                        prompt = llm.edit_kv_cache(
                            original,
                            text,
                            start,
                            end,
                            cache_salt=source_salt,
                            **options,
                        )
                    actual, _ = snapshot(edited_ids, prompt["cache_salt"])
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
                    expected_positions = list(range(start, suffix_start))
                    expected_positions += [
                        p for lo, hi in plan.repair_ranges for p in range(lo, hi)
                    ]
                    item["forward_positions"] = forwards
                    if forwards != expected_positions:
                        raise AssertionError("Unexpected recomputation positions.")
                    if mode in ("full", "all_blocks") or set(plan.repair_blocks) == set(
                        blocks
                    ):
                        item["max_abs_error_vs_fresh"] = compare(actual, fresh)
                    reused = [
                        p
                        for p in range(suffix_start, len(edited_ids))
                        if p // 16 in plan.reuse_blocks
                    ]
                    # Independent complex-number RoPE reference (Qwen2 NeoX).
                    for name, kv in actual.items():
                        torch.testing.assert_close(
                            kv[:start], before[name][:start], atol=0, rtol=0
                        )
                        if not reused:
                            continue
                        rope = model.get_submodule(name.rsplit(".", 1)[0]).rotary_emb
                        if not rope.is_neox_style or rope.rotary_dim != rope.head_size:
                            raise RuntimeError("Test oracle requires full NeoX RoPE.")
                        new = torch.tensor(reused, device=kv.device)
                        old = new - row["shift"]
                        head = rope.head_size
                        src = before[name][old]
                        dst = kv[new]
                        torch.testing.assert_close(
                            dst[..., head:], src[..., head:], atol=0, rtol=0
                        )
                        phase = torch.complex(*rope.cos_sin_cache.float().chunk(2, -1))
                        key = torch.complex(*src[..., :head].float().chunk(2, -1))
                        moved = key * (phase[new] / phase[old])[:, None, :]
                        expected = torch.cat((moved.real, moved.imag), -1)
                        torch.testing.assert_close(
                            dst[..., :head].float(), expected, atol=0.005, rtol=0.002
                        )
                    source_after, _ = snapshot(original, source_salt)
                    compare(source_after, before, exact=True)
                    output = generate(prompt, params)
                    item["generated_ids"] = output.outputs[0].token_ids
                    item["generated_text"] = output.outputs[0].text
                    item["cached_tokens"] = output.num_cached_tokens
                    if len(item["generated_ids"]) != 2:
                        raise AssertionError("Generation did not complete two tokens.")
                    if output.num_cached_tokens != len(edited_ids) - 1:
                        raise AssertionError("Edited prompt did not reuse expected KV.")
                    item["status"] = "passed"
                    persist()
            finally:
                editor.pool.free_blocks(source)
            row["status"] = "passed"
            persist()
        torch.cuda.synchronize()
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    finally:
        llm.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--revision", default="c89bee90d9f811437d9735454613c35b4a3c4dc8"
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--drift-profile", type=Path)
    parser.add_argument("--attention-fraction", type=float)
    parser.add_argument("--attention-query-positions", type=int, nargs="+")
    parser.add_argument("--attention-layers", nargs="+")
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--rtol", type=float, default=0.03)
    args = parser.parse_args()
    if args.attention_fraction is not None and (
        not args.drift_profile or not 0 <= args.attention_fraction <= 1
    ):
        parser.error(
            "Attention capture needs --drift-profile and a fraction in [0, 1]."
        )
    if (
        args.attention_query_positions or args.attention_layers
    ) and args.attention_fraction is None:
        parser.error("Attention query/layer selection needs --attention-fraction.")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    path = args.results_dir / "results.json"
    if path.exists():
        parser.error("Use a new results directory to preserve earlier results.")
    report = {
        "status": "running",
        "cases": [],
        "python": sys.version,
        "platform": platform.platform(),
        "started": datetime.now(UTC).isoformat(),
        "git_head": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "nvidia_smi": command("nvidia-smi"),
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
    }
    root = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            root / "vllm/v1/core/patchkv.py",
            root / "vllm/v1/core/patchkv_profile.py",
            root / "vllm/v1/core/patchkv_attention.py",
            root / "vllm/v1/core/kv_cache_edit.py",
            root / "vllm/v1/worker/kv_cache_edit.py",
            root / "vllm/v1/worker/patchkv_attention.py",
            Path(__file__),
        ]
    }
    persist = lambda: save(path, report)
    persist()
    try:
        validate(args, report, persist)
        report["status"] = "preflight_passed" if args.preflight_only else "passed"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["finished"] = datetime.now(UTC).isoformat()
        persist()
        print(f"Results: {path} ({report['status']})", flush=True)


if __name__ == "__main__":
    main()
