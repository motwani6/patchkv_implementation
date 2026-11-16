# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small format smoke test or real whole-policy quality collection on Colab."""

import argparse
import hashlib
import json
import os
import re
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from suffix_repair_test.patchkv_colab import command, save


def answer_f1(actual, expected):
    def words(text):
        return re.findall(r"\w+", text.lower())

    actual, expected = words(actual), words(expected)
    common = sum((Counter(actual) & Counter(expected)).values())
    return 2 * common / (len(actual) + len(expected)) if common else 0.0


def run(args, report, persist):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5):
        raise RuntimeError("Use a supported Colab CUDA GPU.")
    from vllm import LLM, SamplingParams
    from vllm.v1.core.patchkv import (
        plan_attention_guided_suffix_repair,
        plan_suffix_repair,
    )
    from vllm.v1.core.patchkv_attention import AttentionBudget
    from vllm.v1.core.patchkv_checkpoint import checkpoint_copy_ranges
    from vllm.v1.core.patchkv_precision import (
        PrecisionPolicy,
        QualityMeasurement,
        attention_config_fingerprint,
        repair_policy_fingerprint,
        transport_config_fingerprint,
    )
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
    report.update(
        requested_config=config,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        nvidia_smi=command("nvidia-smi"),
    )
    drift_profile = (
        DriftProfile.load(args.drift_profile) if args.drift_profile else None
    )
    llm = LLM(**config)
    measurements = []
    candidate_records = {}
    try:
        core = llm.llm_engine.engine_core.engine_core
        editor = core.scheduler.kv_cache_manager.editor
        if editor is None:
            raise RuntimeError("Unsupported KV editor configuration.")
        worker = core.model_executor.driver_worker
        model = worker.get_model()
        context = worker.vllm_config.compilation_config.static_forward_context
        names = core.scheduler.kv_cache_manager.kv_cache_config.kv_cache_groups[
            0
        ].layer_names
        tokenizer = llm.get_tokenizer()
        model_key = f"{args.model}@{args.revision};fp16;TRITON_ATTN;maxlen=192"
        if args.tasks:
            tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
        else:
            ids = tokenizer.encode(
                "Remember blue birds in the quiet garden. " * 16,
                add_special_tokens=False,
            )[:83]
            tasks = [
                dict(
                    task_id=label,
                    original_ids=ids,
                    start=19,
                    end=end,
                    replacement_text=text,
                )
                for label, end, text in (
                    ("smoke-insertion", 19, " red green"),
                    ("smoke-deletion", 23, ""),
                    ("smoke-replacement", 21, " red green"),
                )
            ]
        if not tasks or len({task["task_id"] for task in tasks}) != len(tasks):
            raise ValueError(
                "Use unique task IDs and independent tasks for calibration."
            )
        grid = (
            json.loads(args.candidates.read_text(encoding="utf-8"))
            if args.candidates
            else None
        )
        if grid is not None and (not grid or len(grid) > 8):
            raise ValueError(
                "Use a small explicit threshold grid (at most eight candidates)."
            )
        report["collection_kind"] = (
            "whole_policy_quality" if grid else "format_smoke_not_calibration"
        )
        report["effective_config"] = str(worker.vllm_config)

        def snapshot(ids, salt):
            request = editor._request(ids, salt)
            pages = []
            for lo in range(0, len(ids), 16):
                hit = editor.pool.get_cached_block(
                    editor._hash_at(request, min(lo + 16, len(ids))), [0]
                )
                if hit is None:
                    raise AssertionError("Restored KV alias missing.")
                pages.append(hit[0].block_id)
            states = {
                name: torch.cat(
                    [context[name].kv_cache[page].transpose(0, 1) for page in pages]
                )[: len(ids)]
                .detach()
                .cpu()
                .clone()
                for name in names
            }
            return states, pages

        for task in tasks:
            ids = list(task["original_ids"])
            start, end = task["start"], task["end"]
            replacement_text = task["replacement_text"]
            replacement = (
                tokenizer.encode(replacement_text, add_special_tokens=False)
                if replacement_text
                else []
            )
            if not ids or len(ids) > 160 or not 0 <= start <= end <= len(ids):
                raise ValueError(
                    "Use a nonempty <=160-token source and valid edit bounds."
                )
            selected = task.get("repair_suffix_blocks", [])
            plan = plan_suffix_repair(
                len(ids), start, end, len(replacement), 16, selected
            )
            max_tokens = task.get("max_tokens", 4)
            if (
                type(max_tokens) is not int
                or not 1 <= max_tokens <= 16
                or plan.new_length + max_tokens > 192
                or not plan.new_length
            ):
                raise ValueError(
                    "Use 1..16 generated tokens within the 192-token context."
                )
            if grid and (
                not isinstance(task.get("expected_answer"), str)
                or not task["expected_answer"].strip()
            ):
                raise ValueError(
                    "Quality collection requires expected_answer for each real task."
                )
            sampling = SamplingParams(max_tokens=max_tokens, temperature=0)
            source_salt = uuid4().hex
            importance = None
            if grid:
                from vllm.v1.worker.patchkv_attention import (
                    CheckpointAttentionCollector,
                )

                capture = CheckpointAttentionCollector(
                    worker.model_runner,
                    ids,
                    model_key=model_key,
                    checkpoint_id=source_salt,
                    block_size=16,
                    layer_names=[
                        names[i] for i in sorted({0, len(names) // 2, len(names) - 1})
                    ],
                    query_positions=[len(ids) - 1],
                )
                with capture:
                    llm.generate(
                        {"prompt_token_ids": ids, "cache_salt": source_salt},
                        SamplingParams(max_tokens=1, temperature=0),
                        use_tqdm=False,
                    )
                importance = capture.finalize()
            else:
                llm.generate(
                    {"prompt_token_ids": ids, "cache_salt": source_salt},
                    SamplingParams(max_tokens=1, temperature=0),
                    use_tqdm=False,
                )
            if drift_profile is not None:
                budget = AttentionBudget(fraction=args.attention_fraction)
                plan = plan_attention_guided_suffix_repair(
                    ids,
                    start,
                    end,
                    len(replacement),
                    16,
                    drift_profile,
                    importance,
                    budget,
                    model_key=model_key,
                    checkpoint_id=source_salt,
                )
                selected = list(plan.repair_blocks)
            _, source_pages = snapshot(ids, source_salt)
            baseline = llm.create_kv_checkpoint(
                ids,
                model_key=model_key,
                cache_salt=source_salt,
                attention=importance,
                max_cpu_bytes=64 << 20,
            )
            row = dict(task=task, status="running", formats=[])
            report["tasks"].append(row)
            persist()
            try:
                fingerprint = transport_config_fingerprint(
                    baseline.metadata.model_config_json
                )
                attention_fingerprint = (
                    attention_config_fingerprint(
                        baseline.metadata.attention_json, len(ids)
                    )
                    if grid
                    else None
                )
                policies = (
                    [
                        PrecisionPolicy(
                            model_key,
                            16,
                            config_fingerprint=fingerprint,
                            attention_fingerprint=attention_fingerprint,
                            repair_fingerprint=repair_policy_fingerprint(
                                asdict(drift_profile) if drift_profile else None,
                                asdict(
                                    AttentionBudget(fraction=args.attention_fraction)
                                )
                                if drift_profile
                                else None,
                            ),
                            **values,
                        )
                        for values in grid
                    ]
                    if grid
                    else [
                        PrecisionPolicy.experimental_fixed(model_key, 16, fmt)
                        for fmt in ("FP16", "K8/V8", "K8/V4")
                    ]
                )
                for policy in policies:
                    candidate_records[policy.policy_id] = asdict(policy)

                def evict(source_pages=source_pages):
                    if not llm.reset_prefix_cache():
                        raise RuntimeError("Prefix-cache eviction failed.")
                    for name in names:
                        indices = torch.tensor(
                            source_pages, device=context[name].kv_cache.device
                        )
                        context[name].kv_cache.index_fill_(0, indices, float("nan"))
                    worker.synchronize_device()

                evict()
                reference_prompt = llm.edit_kv_cache(
                    ids,
                    replacement_text,
                    start,
                    end,
                    checkpoint=baseline,
                    repair_suffix_blocks=selected,
                )
                reference, _ = snapshot(
                    reference_prompt["prompt_token_ids"], reference_prompt["cache_salt"]
                )
                reference_output = llm.generate(
                    reference_prompt, sampling, use_tqdm=False
                )[0].outputs[0]
                reference_score = (
                    answer_f1(reference_output.text, task["expected_answer"])
                    if grid
                    else 1.0
                )
                row.update(
                    baseline_ids=reference_output.token_ids,
                    baseline_text=reference_output.text,
                    baseline_quality=reference_score,
                    baseline_transfer=baseline.restore_statistics,
                    baseline_payload_bytes=baseline.metadata.payload_bytes,
                )
                for policy in policies:
                    used_before = baseline._session.store.used_bytes
                    encoded = baseline.compress(policy, protected_prefix_tokens=start)
                    try:
                        item = dict(
                            policy_id=policy.policy_id,
                            status="running",
                            quality_calibrated=policy.quality_calibrated,
                            payload_bytes=encoded.metadata.payload_bytes,
                            managed_bytes=baseline._session.store.used_bytes
                            - used_before,
                            precision_tags=json.loads(encoded.metadata.transport_json)[
                                "precision_tags"
                            ],
                        )
                        row["formats"].append(item)
                        persist()
                        evict()
                        tick = time.perf_counter()
                        recovery_options = (
                            dict(
                                drift_profile=drift_profile,
                                attention_budget=AttentionBudget(
                                    fraction=args.attention_fraction
                                ),
                            )
                            if drift_profile is not None
                            else dict(repair_suffix_blocks=selected)
                        )
                        prompt = llm.edit_kv_cache(
                            ids,
                            replacement_text,
                            start,
                            end,
                            checkpoint=encoded,
                            **recovery_options,
                        )
                        elapsed_ms = (time.perf_counter() - tick) * 1000
                        actual, _ = snapshot(
                            prompt["prompt_token_ids"], prompt["cache_salt"]
                        )
                        max_error = 0.0
                        with encoded._session.store.lease(
                            encoded.checkpoint_id
                        ) as reader:
                            for name, kv in actual.items():
                                torch.testing.assert_close(
                                    kv[:start], reference[name][:start], atol=0, rtol=0
                                )
                                max_error = max(
                                    max_error, (kv - reference[name]).abs().max().item()
                                )
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
                                phase = torch.complex(
                                    *rope.cos_sin_cache.float().cpu().chunk(2, -1)
                                )
                                for mapping in checkpoint_copy_ranges(
                                    len(ids), start, end, len(replacement), plan
                                ):
                                    source = reader.read(
                                        name, mapping.source_start, mapping.count
                                    )
                                    old = torch.arange(
                                        mapping.source_start,
                                        mapping.source_start + mapping.count,
                                    )
                                    new = torch.arange(
                                        mapping.destination_start,
                                        mapping.destination_start + mapping.count,
                                    )
                                    torch.testing.assert_close(
                                        kv[new, :, rope.head_size :],
                                        source[..., rope.head_size :],
                                        atol=0,
                                        rtol=0,
                                    )
                                    key = torch.complex(
                                        *source[..., : rope.head_size]
                                        .float()
                                        .chunk(2, -1)
                                    )
                                    moved = key * (phase[new] / phase[old])[:, None, :]
                                    expected = torch.cat((moved.real, moved.imag), -1)
                                    torch.testing.assert_close(
                                        kv[new, :, : rope.head_size].float(),
                                        expected,
                                        atol=0.005,
                                        rtol=0.002,
                                    )
                                if policy.fixed_format == "FP16":
                                    torch.testing.assert_close(
                                        kv, reference[name], atol=0.005, rtol=0.002
                                    )
                        output = llm.generate(prompt, sampling, use_tqdm=False)[
                            0
                        ].outputs[0]
                        score = (
                            answer_f1(output.text, task["expected_answer"])
                            if grid
                            else float(output.token_ids == reference_output.token_ids)
                        )
                        metric = (
                            "answer_token_f1"
                            if grid
                            else "greedy_continuation_agreement_smoke"
                        )
                        measurement = QualityMeasurement(
                            task["task_id"],
                            policy.policy_id,
                            reference_score,
                            score,
                            encoded.restore_statistics["transfer_bytes"],
                            elapsed_ms,
                            metric,
                            str(args.results_dir / "quantized-results.json")
                            + "#"
                            + task["task_id"],
                            "model_execution",
                        )
                        measurements.append(asdict(measurement))
                        item.update(
                            status="passed",
                            transfer=encoded.restore_statistics,
                            max_abs_kv_error_vs_fp16=max_error,
                            quality=score,
                            generated_ids=output.token_ids,
                            generated_text=output.text,
                            restore_ms_single_observation=elapsed_ms,
                        )
                        encoded.metadata.save(
                            args.results_dir / f"metadata-{len(measurements)}.json"
                        )
                        save(
                            args.results_dir / "quality-measurements.json",
                            dict(
                                schema_version=1,
                                candidates=list(candidate_records.values()),
                                measurements=measurements,
                            ),
                        )
                        persist()
                    finally:
                        encoded.release()
            finally:
                baseline.release()
            if llm._patchkv_checkpoint_session.store.used_bytes:
                raise AssertionError(
                    "Checkpoint memory accounting did not return to zero."
                )
            row["status"] = "passed"
            persist()
    finally:
        llm.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--drift-profile", type=Path)
    parser.add_argument("--attention-fraction", type=float, default=0.05)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--revision", default="c89bee90d9f811437d9735454613c35b4a3c4dc8"
    )
    args = parser.parse_args()
    if args.candidates and not args.tasks:
        parser.error("Quality collection requires real --tasks with expected answers.")
    if args.drift_profile and (
        not args.candidates or not 0 <= args.attention_fraction <= 1
    ):
        parser.error(
            "Guided quality collection requires candidates and a fraction in [0,1]."
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    args.results_dir.mkdir(parents=True, exist_ok=False)
    path = args.results_dir / "quantized-results.json"
    root = Path(__file__).resolve().parents[1]
    report = dict(
        status="running",
        tasks=[],
        git_head=command("git", "rev-parse", "HEAD"),
        git_status=command("git", "status", "--short"),
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                *root.glob("vllm/v1/core/patchkv*.py"),
                root / "vllm/entrypoints/patchkv.py",
                root / "vllm/v1/worker/patchkv_checkpoint.py",
            ]
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


if __name__ == "__main__":
    main()
