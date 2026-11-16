# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect real exact-prefill drift on Colab or fit profiles using CPU only."""

import argparse
import json
import math
import os
import traceback
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from suffix_repair_test.patchkv_colab import command, save
from suffix_repair_test.patchkv_profile_tools import load_profile_module

p = load_profile_module()


def collect(args):
    if (
        args.block_size <= 0
        or args.block_size % 16
        or not math.isfinite(args.tau)
        or not 0 <= args.tau <= 2
        or not math.isfinite(args.gamma)
        or not 0 < args.gamma <= 1
        or not math.isfinite(args.aggregate_quantile)
        or not 0 < args.aggregate_quantile <= 1
    ):
        raise ValueError("Invalid drift settings or unsupported Triton block size.")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("collect-gpu requires a CUDA runtime; use Colab.")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    from vllm import LLM, SamplingParams

    raw = json.loads(args.pairs.read_text(encoding="utf-8"))
    pairs = []
    for row in raw:
        if "edited_ids" in row:
            if set(row) != {"pair_id", "task_id", "original_ids", "edited_ids"}:
                raise ValueError("Do not mix full contexts and explicit edit spans.")
            pairs.append(p.ProfilePair.from_contexts(**row))
        else:
            pairs.append(
                p.ProfilePair(
                    **{
                        **row,
                        "original_ids": tuple(row["original_ids"]),
                        "replacement_ids": tuple(row["replacement_ids"]),
                    }
                )
            )
    if not pairs or len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("Need nonempty input with unique pair IDs.")
    if any(
        not pair.original_ids
        or not pair.edited_ids
        or max(len(pair.original_ids), len(pair.edited_ids)) + 1 > args.max_model_len
        for pair in pairs
    ):
        raise ValueError(
            "Paired contexts must be nonempty and fit max_model_len plus decode."
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    observations = []
    metadata = {
        "status": "running",
        "pairs_completed": 0,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "git_head": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "nvidia_smi": command("nvidia-smi"),
        "model": args.model,
        "revision": args.revision,
        "pairs": raw,
    }
    report_path = args.output_dir / "collection.json"
    save(report_path, metadata)
    llm = None
    try:
        llm = LLM(
            model=args.model,
            revision=args.revision,
            enforce_eager=True,
            dtype="float16",
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            max_num_batched_tokens=32,
            block_size=args.block_size,
            enable_prefix_caching=True,
            kv_cache_memory_bytes=32 << 20,
            gpu_memory_utilization=0.6,
            attention_config={"backend": "TRITON_ATTN"},
        )
        core = llm.llm_engine.engine_core.engine_core
        manager = core.scheduler.kv_cache_manager
        editor = manager.editor
        if editor is None:
            raise RuntimeError("Unsupported configuration: KV editor unavailable.")
        worker = core.model_executor.driver_worker
        context = worker.vllm_config.compilation_config.static_forward_context
        names = manager.kv_cache_config.kv_cache_groups[0].layer_names
        indices = sorted({0, len(names) // 2, len(names) - 1})
        layers = tuple(names[i] for i in indices)
        model_key = (
            f"{args.model}@{args.revision};fp16;TRITON_ATTN;maxlen={args.max_model_len}"
        )
        cfg = p.DriftCalibration(
            model_key,
            layers,
            args.block_size,
            args.tau,
            args.gamma,
            args.aggregate_quantile,
        )
        metadata["config"] = asdict(cfg)
        metadata["effective_config"] = str(worker.vllm_config)
        params = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)

        def exact_prefill(ids):
            salt = uuid4().hex
            output = llm.generate(
                {"prompt_token_ids": list(ids), "cache_salt": salt},
                params,
                use_tqdm=False,
            )[0]
            if output.num_cached_tokens != 0:
                raise AssertionError("Profiling prefill unexpectedly reused KV.")
            request = editor._request(list(ids), salt)
            pages = []
            for pos in range(0, len(ids), args.block_size):
                end = min(pos + args.block_size, len(ids))
                hit = editor.pool.get_cached_block(editor._hash_at(request, end), [0])
                if hit is None:
                    raise RuntimeError("Profiling KV was evicted before snapshot.")
                pages.append(hit[0].block_id)
            values = {}
            for name in layers:
                cache = context[name].kv_cache
                layer = context[name]
                if (
                    cache.ndim != 4
                    or cache.shape[1] != layer.num_kv_heads
                    or cache.shape[2] != args.block_size
                    or cache.shape[-1] % 2
                ):
                    raise RuntimeError(
                        f"Unsupported profiling cache layout: {name} {cache.shape}"
                    )
                head = cache.shape[-1] // 2
                values[name] = (
                    torch.cat(
                        [cache[page].transpose(0, 1)[..., head:] for page in pages]
                    )[: len(ids)]
                    .detach()
                    .cpu()
                    .clone()
                )
            return values

        for pair in pairs:
            observations.append(p.profile_pair(pair, exact_prefill, cfg))
            p.save_observations(args.output_dir / "observations.json", observations)
            metadata["pairs_completed"] = len(observations)
            metadata["last_dirty_window"] = observations[-1].dirty_window
            save(report_path, metadata)
        metadata["status"] = "passed"
    except Exception:
        metadata["status"] = "failed"
        metadata["traceback"] = traceback.format_exc()
        raise
    finally:
        save(report_path, metadata)
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


def fit(args):
    samples = p.load_observations(args.observations)
    training, calibration = p.split_by_task(
        samples, args.calibration_fraction, args.seed
    )
    profile = p.fit_drift_profile(training, calibration, args.quantile)
    summary = {
        "status": "fit_completed",
        "training_pairs": len(training),
        "calibration_pairs": len(calibration),
        "training_tasks": len(profile.training_tasks),
        "calibration_tasks": len(profile.calibration_tasks),
        "residual_margin": profile.residual_margin,
        "full_suffix_fallback": profile.residual_margin is None,
        "training_pinball_loss": profile.training_pinball_loss,
        "fit_sweeps": profile.fit_sweeps,
    }
    if args.validation:
        validation = p.load_observations(args.validation)
        seen = set(profile.training_tasks) | set(profile.calibration_tasks)
        seen_pairs = {s.pair_id for s in samples}
        if (
            not validation
            or any(
                s.config != profile.config
                or s.task_id in seen
                or s.pair_id in seen_pairs
                for s in validation
            )
            or len({s.pair_id for s in validation}) != len(validation)
        ):
            raise ValueError(
                "Validation must use the same config and wholly unseen tasks."
            )
        predictions = [profile.predict(*s.lengths) for s in validation]
        summary["validation"] = {
            "pairs": len(validation),
            "boundary_coverage": sum(
                d >= s.dirty_window for d, s in zip(predictions, validation)
            )
            / len(validation),
            "predicted_tokens": predictions,
            "oracle_tokens": [s.dirty_window for s in validation],
            "pair_ids": [s.pair_id for s in validation],
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    profile.save(args.output_dir / "profile.json")
    save(args.output_dir / "fit_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    collector = sub.add_parser("collect-gpu", help="Colab-only paired exact prefills")
    collector.add_argument("--pairs", type=Path, required=True)
    collector.add_argument("--output-dir", type=Path, required=True)
    collector.add_argument("--tau", type=float, required=True)
    collector.add_argument("--gamma", type=float, required=True)
    collector.add_argument("--aggregate-quantile", type=float, default=0.9)
    collector.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    collector.add_argument(
        "--revision", default="c89bee90d9f811437d9735454613c35b4a3c4dc8"
    )
    collector.add_argument("--block-size", type=int, default=16)
    collector.add_argument("--max-model-len", type=int, default=192)
    fitter = sub.add_parser("fit", help="CPU-only fit; no torch or vLLM runtime import")
    fitter.add_argument("--observations", type=Path, required=True)
    fitter.add_argument("--output-dir", type=Path, required=True)
    fitter.add_argument("--validation", type=Path)
    fitter.add_argument("--quantile", type=float, default=0.95)
    fitter.add_argument("--calibration-fraction", type=float, default=0.25)
    fitter.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    (collect if args.mode == "collect-gpu" else fit)(args)


if __name__ == "__main__":
    main()
