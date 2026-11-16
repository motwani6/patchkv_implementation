# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small held-out Colab benchmark; real measurements only, no local execution."""

import argparse
import hashlib
import json
import os
import statistics
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from suffix_repair_test.patchkv_colab import save
from suffix_repair_test.patchkv_quantized_colab import answer_f1


def summarize(rows):
    """Aggregate completed paired observations, retaining failures separately."""
    result = {}
    references = {
        (r["task_id"], r["repetition"]): r
        for r in rows
        if r.get("status") == "passed" and r["mode"] == "full"
    }
    for mode in sorted({r["mode"] for r in rows}):
        group = [r for r in rows if r["mode"] == mode and r.get("status") == "passed"]
        if not group:
            continue
        item = {"observations": len(group)}
        for key in (
            "recovery_ms",
            "generation_ms",
            "resume_to_response_ms",
            "gpu_peak_allocated_bytes",
            "cpu_payload_bytes",
            "transfer_bytes",
            "answer_token_f1",
            "continuation_agreement",
        ):
            item[key + "_mean"] = statistics.mean(r[key] for r in group)
            item[key + "_median"] = statistics.median(r[key] for r in group)
        paired = [
            references[(r["task_id"], r["repetition"])]["recovery_ms"]
            / r["recovery_ms"]
            for r in group
            if (r["task_id"], r["repetition"]) in references and r["recovery_ms"] > 0
        ]
        item["paired_recovery_speedup_median"] = (
            statistics.median(paired) if paired else None
        )
        result[mode] = item
    return result


def run(args, report, persist):
    from vllm.v1.core.patchkv_attention import AttentionBudget
    from vllm.v1.core.patchkv_precision import (
        PrecisionPolicy,
        repair_policy_fingerprint,
    )
    from vllm.v1.core.patchkv_profile import DriftProfile

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    profile = DriftProfile.load(args.drift_profile)
    policy = PrecisionPolicy.load(args.precision_policy)
    budget = AttentionBudget(fraction=args.attention_fraction)
    expected_scope = repair_policy_fingerprint(asdict(profile), asdict(budget))
    if not policy.quality_calibrated or policy.repair_fingerprint != expected_scope:
        raise ValueError(
            "Use a quality-calibrated policy for this exact drift/attention policy."
        )
    used = set(
        profile.training_tasks
        + profile.calibration_tasks
        + policy.calibration_tasks
        + policy.validation_tasks
    )
    if (
        not tasks
        or len({t["task_id"] for t in tasks}) != len(tasks)
        or used & {t["task_id"] for t in tasks}
    ):
        raise ValueError(
            "Benchmark tasks must be unique and held out from fitting/calibration."
        )
    for task in tasks:
        if (
            not isinstance(task.get("expected_answer"), str)
            or not task["expected_answer"].strip()
        ):
            raise ValueError("Held-out quality tasks require expected_answer.")
    model_key = f"{args.model}@{args.revision};fp16;TRITON_ATTN;maxlen=192"
    if policy.model_key != model_key or profile.config.model_key != model_key:
        raise ValueError(
            "Frozen profiles must match the requested model/revision/configuration."
        )

    import torch

    from vllm import LLM, SamplingParams
    from vllm.v1.worker.patchkv_attention import CheckpointAttentionCollector

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required; run the consolidated Colab preflight first."
        )
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
    report.update(
        requested_config=config,
        gpu=torch.cuda.get_device_name(),
        timing_definition=(
            "Synchronized recovery including planning; "
            "nonstreamed generation separately"
        ),
    )
    llm = LLM(**config)
    try:
        core = llm.llm_engine.engine_core.engine_core
        worker = core.model_executor.driver_worker
        context = worker.vllm_config.compilation_config.static_forward_context
        editor = core.scheduler.kv_cache_manager.editor
        if editor is None:
            raise RuntimeError("Unsupported dense single-GPU editor configuration.")
        names = core.scheduler.kv_cache_manager.kv_cache_config.kv_cache_groups[
            0
        ].layer_names
        tokenizer = llm.get_tokenizer()
        report["effective_config"] = str(worker.vllm_config)
        storages = {
            context[n].kv_cache.untyped_storage().data_ptr(): context[n]
            .kv_cache.untyped_storage()
            .nbytes()
            for n in names
        }
        report["gpu_kv_capacity_bytes"] = sum(storages.values())

        def snapshot(prompt):
            tokens = prompt["prompt_token_ids"]
            request = editor._request(tokens, prompt["cache_salt"])
            pages = []
            for lo in range(0, len(tokens), 16):
                hit = editor.pool.get_cached_block(
                    editor._hash_at(request, min(lo + 16, len(tokens))), [0]
                )
                if hit is None:
                    raise RuntimeError("Recovered cache mapping missing.")
                pages.append(hit[0].block_id)
            return {
                name: torch.cat(
                    [context[name].kv_cache[page].transpose(0, 1) for page in pages]
                )[: len(tokens)]
                .detach()
                .float()
                .cpu()
                for name in names
            }

        # One unmeasured model warmup; checkpoint operations are measured cold per task.
        llm.generate(
            "Warm up.", SamplingParams(max_tokens=1, temperature=0), use_tqdm=False
        )
        for task_index, task in enumerate(tasks):
            ids, start, end = task["original_ids"], task["start"], task["end"]
            text = task["replacement_text"]
            replacement = (
                tokenizer.encode(text, add_special_tokens=False) if text else []
            )
            max_tokens = task.get("max_tokens", 4)
            if (
                not ids
                or len(ids) > 160
                or not 0 <= start <= end <= len(ids)
                or type(max_tokens) is not int
                or not 1 <= max_tokens <= 16
                or not 0 < len(ids) - end + start + len(replacement) <= 192 - max_tokens
            ):
                raise ValueError(
                    "Use <=160-token sources and <=16-token outputs within 192 tokens."
                )
            salt = uuid4().hex
            capture = CheckpointAttentionCollector(
                worker.model_runner,
                ids,
                model_key=model_key,
                checkpoint_id=salt,
                block_size=16,
                layer_names=[
                    names[i] for i in sorted({0, len(names) // 2, len(names) - 1})
                ],
                query_positions=[len(ids) - 1],
            )
            with capture:
                llm.generate(
                    {"prompt_token_ids": ids, "cache_salt": salt},
                    SamplingParams(max_tokens=1, temperature=0),
                    use_tqdm=False,
                )
            checkpoint = llm.create_kv_checkpoint(
                ids,
                model_key=model_key,
                cache_salt=salt,
                attention=capture.finalize(),
                max_cpu_bytes=64 << 20,
            )
            encoded = None
            try:
                pages = []
                request = editor._request(ids, salt)
                for lo in range(0, len(ids), 16):
                    hit = editor.pool.get_cached_block(
                        editor._hash_at(request, min(lo + 16, len(ids))), [0]
                    )
                    if hit is None:
                        raise RuntimeError("Original checkpoint source pages missing.")
                    pages.append(hit[0].block_id)
                source_charged_bytes = checkpoint._session.store.used_bytes
                encoded = checkpoint.compress(policy, protected_prefix_tokens=start)
                encoded_charged_bytes = (
                    checkpoint._session.store.used_bytes - source_charged_bytes
                )
                options = dict(
                    direct={},
                    full={"recompute_suffix": True},
                    manual={
                        "repair_suffix_blocks": task.get("repair_suffix_blocks", [])
                    },
                    drift={"drift_profile": profile},
                    attention={"drift_profile": profile, "attention_budget": budget},
                    quantized={"drift_profile": profile, "attention_budget": budget},
                )
                reference_ids = None
                for repetition in range(args.repetitions):
                    # Full reference first; rotate other modes to reduce order bias.
                    modes = ["direct", "manual", "drift", "attention", "quantized"]
                    offset = (task_index + repetition) % len(modes)
                    for mode in ["full", *modes[offset:], *modes[:offset]]:
                        handle = encoded if mode == "quantized" else checkpoint
                        row = dict(
                            task_id=task["task_id"],
                            repetition=repetition,
                            mode=mode,
                            status="running",
                        )
                        report["observations"].append(row)
                        persist()
                        if not llm.reset_prefix_cache():
                            raise RuntimeError("Prefix cache could not be evicted.")
                        for name in names:
                            cache = context[name].kv_cache
                            cache.index_fill_(
                                0,
                                torch.tensor(pages, device=cache.device),
                                float("nan"),
                            )
                        worker.synchronize_device()
                        before = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        tick = time.perf_counter()
                        prompt = llm.edit_kv_cache(
                            ids, text, start, end, checkpoint=handle, **options[mode]
                        )
                        worker.synchronize_device()
                        restored = time.perf_counter()
                        recovery_peak = torch.cuda.max_memory_allocated()
                        recovery_reserved = torch.cuda.max_memory_reserved()
                        states = snapshot(prompt)
                        if mode == "full":
                            reference_states = states
                        squared, elements, max_error = 0.0, 0, 0.0
                        for name, state in states.items():
                            if not torch.isfinite(state).all():
                                raise AssertionError(
                                    "Recovered cache contains NaN/Inf."
                                )
                            delta = state - reference_states[name]
                            squared += delta.square().sum().item()
                            elements += delta.numel()
                            max_error = max(max_error, delta.abs().max().item())
                        # Exclude the diagnostic CPU KV snapshot from generation timing.
                        worker.synchronize_device()
                        torch.cuda.reset_peak_memory_stats()
                        generation_start = time.perf_counter()
                        output = llm.generate(
                            prompt,
                            SamplingParams(max_tokens=max_tokens, temperature=0),
                            use_tqdm=False,
                        )[0].outputs[0]
                        worker.synchronize_device()
                        done = time.perf_counter()
                        if mode == "full":
                            reference_ids = list(output.token_ids)
                        row.update(
                            status="passed",
                            recovery_ms=1000 * (restored - tick),
                            generation_ms=1000 * (done - generation_start),
                            resume_to_response_ms=1000
                            * (restored - tick + done - generation_start),
                            gpu_peak_allocated_bytes=max(
                                recovery_peak, torch.cuda.max_memory_allocated()
                            ),
                            gpu_peak_delta_bytes=max(
                                0,
                                max(recovery_peak, torch.cuda.max_memory_allocated())
                                - before,
                            ),
                            gpu_peak_reserved_bytes=max(
                                recovery_reserved, torch.cuda.max_memory_reserved()
                            ),
                            cpu_payload_bytes=handle.metadata.payload_bytes,
                            cpu_charged_bytes=(
                                encoded_charged_bytes
                                if mode == "quantized"
                                else source_charged_bytes
                            ),
                            kv_rmse_vs_full=(squared / elements) ** 0.5,
                            kv_max_abs_error_vs_full=max_error,
                            transfer_bytes=handle.restore_statistics.get(
                                "transfer_bytes", 0
                            ),
                            transfer=handle.restore_statistics,
                            continuation_agreement=float(
                                list(output.token_ids) == reference_ids
                            ),
                            answer_token_f1=answer_f1(
                                output.text, task["expected_answer"]
                            ),
                            generated_ids=list(output.token_ids),
                            generated_text=output.text,
                        )
                        report["summary"] = summarize(report["observations"])
                        persist()
            finally:
                if encoded is not None:
                    encoded.release()
                checkpoint.release()
            if llm._patchkv_checkpoint_session.store.used_bytes:
                raise AssertionError("Checkpoint ownership leaked after task cleanup.")
    finally:
        llm.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tasks", "drift-profile", "precision-policy", "results-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--revision", default="c89bee90d9f811437d9735454613c35b4a3c4dc8"
    )
    parser.add_argument("--attention-fraction", type=float, default=0.05)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if not os.environ.get("COLAB_RELEASE_TAG") or not 1 <= args.repetitions <= 3:
        parser.error("Run on Colab only, with 1..3 repetitions.")
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_USE_V2_MODEL_RUNNER="0")
    args.results_dir.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running",
        observations=[],
        artifact_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.tasks, args.drift_profile, args.precision_policy)
        },
    )
    persist = lambda: save(args.results_dir / "benchmark-results.json", report)
    persist()
    try:
        run(args, report, persist)
        report["status"] = "passed"
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
