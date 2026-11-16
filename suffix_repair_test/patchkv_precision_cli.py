# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only policy fitting from saved whole-request quality measurements."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from suffix_repair_test.patchkv_colab import save
from suffix_repair_test.patchkv_profile_tools import load_transport_support

p = load_transport_support("patchkv_precision")


def fit(args):
    data = json.loads(args.measurements.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported quality measurement schema.")
    candidates = [p.PrecisionPolicy.from_dict(row) for row in data["candidates"]]
    rows = [p.QualityMeasurement(**row) for row in data["measurements"]]
    index = {(row.policy_id, row.task_id): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate task/policy measurements.")

    def evaluate(policy, task):
        if (policy.policy_id, task) not in index:
            raise ValueError(
                f"Missing whole-policy quality measurement for task {task}."
            )
        return index[policy.policy_id, task]

    result, used = p.calibrate_precision_policy(
        candidates,
        args.calibration_tasks,
        args.validation_tasks,
        evaluate,
        args.max_quality_loss,
        args.latency_weight,
        args.min_baseline_score,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result.save(args.output_dir / "precision-policy.json")
    report = dict(
        status="quality_calibrated" if result.quality_calibrated else "uncalibrated",
        policy_id=result.policy_id,
        quality_calibrated=result.quality_calibrated,
        evidence_kinds=sorted({row.evidence_kind for row in used}),
        measurements_used=[asdict(row) for row in used],
        input_sha256=hashlib.sha256(args.measurements.read_bytes()).hexdigest(),
        max_quality_loss=args.max_quality_loss,
        latency_weight=args.latency_weight,
        min_baseline_score=args.min_baseline_score,
    )
    save(args.output_dir / "calibration-summary.json", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "measurements_used"},
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--calibration-tasks", nargs="+", required=True)
    parser.add_argument("--validation-tasks", nargs="+", required=True)
    parser.add_argument("--max-quality-loss", type=float, required=True)
    parser.add_argument("--latency-weight", type=float, default=0)
    parser.add_argument("--min-baseline-score", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
