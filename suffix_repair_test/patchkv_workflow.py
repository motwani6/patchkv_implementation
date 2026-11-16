# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consolidated Colab workflow. Default is plan-only: never allocates a GPU."""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from suffix_repair_test.patchkv_colab import command, save

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "c89bee90d9f811437d9735454613c35b4a3c4dc8"


def validate_edit_tasks(rows):
    for row in rows:
        ids = row.get("original_ids")
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= 160
            or any(type(token) is not int or token < 0 for token in ids)
            or type(row.get("start")) is not int
            or type(row.get("end")) is not int
            or not 0 <= row["start"] <= row["end"] <= len(ids)
            or not isinstance(row.get("replacement_text"), str)
            or not isinstance(row.get("expected_answer"), str)
            or not row["expected_answer"].strip()
            or type(row.get("max_tokens", 4)) is not int
            or not 1 <= row.get("max_tokens", 4) <= 16
        ):
            raise ValueError(
                "Invalid real edit task: check IDs, bounds, answer and token limits."
            )
        selected = row.get("repair_suffix_blocks", [])
        if not isinstance(selected, list) or any(
            type(b) is not int or b < 0 for b in selected
        ):
            raise ValueError(
                "Manual repair blocks must be nonnegative integer indices."
            )


def validate_pairs(rows):
    from suffix_repair_test.patchkv_profile_tools import load_profile_module

    p = load_profile_module()
    pairs = [
        p.ProfilePair.from_contexts(**row)
        if "edited_ids" in row
        else p.ProfilePair(**row)
        for row in rows
    ]
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("Drift pair IDs must be unique.")
    if any(
        not pair.original_ids
        or not pair.edited_ids
        or max(len(pair.original_ids), len(pair.edited_ids)) > 191
        for pair in pairs
    ):
        raise ValueError("Drift contexts must fit 192 tokens including generation.")


def build_plan(manifest, base, output):
    stages, inputs, warnings = [], {}, []
    model = [
        "--model",
        manifest.get("model", MODEL),
        "--revision",
        manifest.get("revision", REVISION),
    ]

    def input_file(value):
        path = (base / value).resolve()
        if not path.is_file():
            raise ValueError(f"Required input does not exist: {path}")
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def load_tasks(value):
        path = input_file(value)
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Expected a nonempty task list: {path}")
        tasks = {row["task_id"] for row in rows}
        if any(not isinstance(task, str) or not task for task in tasks):
            raise ValueError("Task IDs must be stable nonempty strings.")
        return path, rows, tasks

    def add(name, module, args, gpu=True):
        stages.append(
            dict(
                name=name,
                gpu=gpu,
                command=[
                    sys.executable,
                    "-m",
                    "suffix_repair_test." + module,
                    *map(str, args),
                ],
            )
        )

    add(
        "00-environment",
        "patchkv_environment",
        ["--gpu", "--output", output / "environment.json"],
    )
    add("01-resident", "patchkv_colab", model + ["--results-dir", output / "resident"])
    add(
        "02-checkpoints",
        "patchkv_checkpoint_colab",
        model + ["--results-dir", output / "checkpoints"],
    )
    add(
        "03-codec-smoke",
        "patchkv_quantized_colab",
        model + ["--results-dir", output / "codec-smoke"],
    )
    seen = set()
    drift_path = output / "drift-fit/profile.json"
    policy_path = output / "precision-fit/precision-policy.json"
    if "drift" in manifest:
        drift = manifest["drift"]
        pairs, rows, seen = load_tasks(drift["pairs"])
        validate_pairs(rows)
        attention_fraction = drift.get("attention_fraction", 0.05)
        if not 0 <= attention_fraction <= 1:
            raise ValueError("Attention fraction must be in [0,1].")
        if not (0 <= drift["tau"] <= 2 and 0 < drift["gamma"] <= 1):
            raise ValueError("Invalid explicit drift tau/gamma.")
        quantile, fraction = (
            drift.get("quantile", 0.95),
            drift.get("calibration_fraction", 0.25),
        )
        if not 0.5 < quantile < 1 or not 0 < fraction < 1 or len(seen) < 2:
            raise ValueError(
                "Need valid quantile/fraction and at least two drift tasks."
            )
        count = min(len(seen) - 1, max(1, math.ceil(len(seen) * fraction)))
        if math.ceil((count + 1) * quantile) > count:
            warnings.append(
                "Insufficient drift calibration tasks: predictions will fall back "
                "to full suffix repair."
            )
        collect = model + ["--tau", drift["tau"], "--gamma", drift["gamma"]]
        add(
            "04-drift-collect",
            "patchkv_profile_cli",
            [
                "collect-gpu",
                "--pairs",
                pairs,
                "--output-dir",
                output / "drift",
                *collect,
            ],
        )
        fit = [
            "fit",
            "--observations",
            output / "drift/observations.json",
            "--output-dir",
            output / "drift-fit",
            "--quantile",
            quantile,
            "--calibration-fraction",
            fraction,
            "--seed",
            drift.get("seed", 0),
        ]
        if "validation_pairs" in drift:
            path, validation_rows, validation = load_tasks(drift["validation_pairs"])
            validate_pairs(validation_rows)
            if seen & validation:
                raise ValueError("Drift validation tasks overlap profiling tasks.")
            seen |= validation
            add(
                "05-drift-validation-collect",
                "patchkv_profile_cli",
                [
                    "collect-gpu",
                    "--pairs",
                    path,
                    "--output-dir",
                    output / "drift-validation",
                    *collect,
                ],
            )
            fit += ["--validation", output / "drift-validation/observations.json"]
        add("06-drift-fit", "patchkv_profile_cli", fit, gpu=False)
        add(
            "07-guided-smoke",
            "patchkv_colab",
            model
            + [
                "--drift-profile",
                drift_path,
                "--attention-fraction",
                drift.get("attention_fraction", 0.05),
                "--results-dir",
                output / "guided-smoke",
            ],
        )
    if "precision" in manifest:
        precision = manifest["precision"]
        if (
            not 0 <= precision["max_quality_loss"] <= 1
            or not 0 <= precision.get("min_baseline_score", 0.5) <= 1
        ):
            raise ValueError("Quality acceptance thresholds must be in [0,1].")
        tasks_path, rows, tasks = load_tasks(precision["tasks"])
        validate_edit_tasks(rows)
        if len(tasks) != len(rows) or seen & tasks:
            raise ValueError(
                "Precision tasks must be unique and independent of drift data."
            )
        seen |= tasks
        calibration, validation = (
            precision["calibration_tasks"],
            precision["validation_tasks"],
        )
        if (
            len(calibration) < 2
            or len(validation) < 2
            or set(calibration) & set(validation)
            or len(set(calibration + validation)) != len(calibration + validation)
            or set(calibration + validation) != tasks
        ):
            raise ValueError(
                "Precision tasks need disjoint, exhaustive "
                "calibration/validation lists."
            )
        grid_path = input_file(precision["candidates"])
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        if not isinstance(grid, list) or not 1 <= len(grid) <= 8:
            raise ValueError("Supply 1..8 explicit precision threshold candidates.")
        for row in grid:
            if set(row) != {"alpha", "tau_low", "tau_high", "v4_error_limit"} or not (
                all(math.isfinite(value) for value in row.values())
                and 0 <= row["alpha"] <= 1
                and 0 <= row["tau_low"] < row["tau_high"]
                and row["v4_error_limit"] >= 0
            ):
                raise ValueError("Invalid precision candidate grid.")
        if any(
            not isinstance(row.get("expected_answer"), str)
            or not row["expected_answer"].strip()
            for row in rows
        ):
            raise ValueError("Real precision tasks require expected answers.")
        add(
            "08-quality-collect",
            "patchkv_quantized_colab",
            model
            + [
                "--tasks",
                tasks_path,
                "--candidates",
                grid_path,
                "--results-dir",
                output / "quality",
                *(
                    [
                        "--drift-profile",
                        drift_path,
                        "--attention-fraction",
                        manifest["drift"].get("attention_fraction", 0.05),
                    ]
                    if "drift" in manifest
                    else []
                ),
            ],
        )
        add(
            "09-precision-fit",
            "patchkv_precision_cli",
            [
                "--measurements",
                output / "quality/quality-measurements.json",
                "--calibration-tasks",
                *calibration,
                "--validation-tasks",
                *validation,
                "--max-quality-loss",
                precision["max_quality_loss"],
                "--min-baseline-score",
                precision.get("min_baseline_score", 0.5),
                "--output-dir",
                output / "precision-fit",
            ],
            gpu=False,
        )
    if "benchmark" in manifest:
        if "drift" not in manifest or "precision" not in manifest:
            raise ValueError(
                "All-mode benchmark requires drift and precision fitting stages."
            )
        benchmark = manifest["benchmark"]
        tasks_path, rows, tasks = load_tasks(benchmark["tasks"])
        validate_edit_tasks(rows)
        if (
            benchmark.get("attention_fraction", attention_fraction)
            != attention_fraction
        ):
            raise ValueError(
                "Benchmark attention policy must match quality calibration."
            )
        if len(rows) != len(tasks) or seen & tasks:
            raise ValueError(
                "Benchmark tasks must be unique and disjoint from "
                "all fitting/validation tasks."
            )
        repetitions = benchmark.get("repetitions", 1)
        if type(repetitions) is not int or not 1 <= repetitions <= 3:
            raise ValueError("Initial benchmark preparation permits 1..3 repetitions.")
        add(
            "10-heldout-benchmark",
            "patchkv_benchmark",
            model
            + [
                "--tasks",
                tasks_path,
                "--drift-profile",
                drift_path,
                "--precision-policy",
                policy_path,
                "--attention-fraction",
                attention_fraction,
                "--repetitions",
                repetitions,
                "--results-dir",
                output / "benchmark",
            ],
        )
    return dict(
        schema_version=1,
        status="planned_not_executed",
        stages=stages,
        input_sha256=inputs,
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for pattern in (
                "suffix_repair_test/patchkv*.py",
                "vllm/v1/core/patchkv*.py",
                "vllm/v1/worker/patchkv*.py",
                "vllm/entrypoints/patchkv.py",
                "vllm/entrypoints/llm.py",
                "vllm/v1/core/kv_cache_edit.py",
                "vllm/v1/worker/kv_cache_edit.py",
                "vllm/v1/engine/core.py",
            )
            for path in ROOT.glob(pattern)
        },
        warnings=warnings,
        gpu_model_loads=sum(stage["gpu"] for stage in stages) - 1,
    )


def execute_plan(
    plan, output, total_timeout=1200, stage_timeout=600, run=subprocess.run
):
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "workflow-results.json"
    if summary_path.exists():
        raise ValueError("Execution results already exist; use a new directory.")
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    for relative, digest in plan.get("source_sha256", {}).items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Implementation changed after plan construction.")
    for value, digest in plan["input_sha256"].items():
        path = Path(value)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("An input changed after plan construction.")
        shutil.copyfile(path, inputs / (digest[:12] + "-" + path.name))
    environment = dict(
        os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_USE_V2_MODEL_RUNNER="0"
    )
    summary = dict(
        status="running",
        stages=[],
        result_scope=(
            "Process completion; consult artifacts for measured correctness/quality"
        ),
        git_head=command("git", "rev-parse", "HEAD"),
        git_status=command("git", "status", "--short"),
    )
    start = time.monotonic()
    save(summary_path, summary)
    try:
        for stage in plan["stages"]:
            remaining = total_timeout - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError(
                    "Workflow time limit reached; no more GPU stages will start."
                )
            record = dict(
                name=stage["name"], status="running", command=stage["command"]
            )
            summary["stages"].append(record)
            save(summary_path, summary)
            tick = time.monotonic()
            with (logs / (stage["name"] + ".log")).open("w", encoding="utf-8") as log:
                result = run(
                    stage["command"],
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=min(stage_timeout, remaining),
                    check=False,
                )
            record.update(
                elapsed_seconds=time.monotonic() - tick,
                returncode=result.returncode,
                status="passed" if result.returncode == 0 else "failed",
            )
            if result.returncode:
                raise RuntimeError(
                    f"Stage {stage['name']} failed; inspect its log. "
                    "No retries/installations were attempted."
                )
            save(summary_path, summary)
        summary["status"] = "passed"
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        if summary["stages"] and summary["stages"][-1]["status"] == "running":
            summary["stages"][-1]["status"] = "failed"
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - start
        save(summary_path, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--execute", action="store_true", help="Explicitly execute on Colab only"
    )
    parser.add_argument("--total-timeout", type=int, default=1200)
    parser.add_argument("--stage-timeout", type=int, default=600)
    args = parser.parse_args()
    if args.total_timeout <= 0 or args.stage_timeout <= 0:
        parser.error("Time limits must be positive.")
    manifest = (
        json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else {}
    )
    output = args.results_dir.resolve()
    plan = build_plan(
        manifest, args.manifest.resolve().parent if args.manifest else ROOT, output
    )
    output.mkdir(parents=True, exist_ok=True)
    save(output / "workflow-plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if not os.environ.get("COLAB_RELEASE_TAG"):
        parser.error("Execution requires Colab; local plan-only mode is available.")
    execute_plan(plan, output, args.total_timeout, args.stage_timeout)


if __name__ == "__main__":
    main()
