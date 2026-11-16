# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen attention/error precision rules calibrated against measured quality."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path


def transport_config_fingerprint(model_config_json: str) -> str:
    config = json.loads(model_config_json)
    # RoPE sample rows depend on source length; full HF RoPE configuration and
    # revision remain bound while layer names, shapes, and dtypes are retained.
    for layer in config.get("layers", []):
        layer.pop("rope_signature", None)
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def attention_config_fingerprint(attention_json: str, token_count: int) -> str:
    data = json.loads(attention_json)
    layers = [
        {key: value for key, value in layer.items() if key != "rope_signature"}
        for layer in data["layers"]
    ]
    config = dict(
        scoring=data["scoring"],
        schema_version=data["schema_version"],
        layers=layers,
        query_offsets=[position - token_count for position in data["query_positions"]],
    )
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def repair_policy_fingerprint(profile=None, budget=None, mode="manual"):
    if budget is not None and profile is None:
        raise ValueError("Attention repair binding needs a drift profile.")
    configuration = dict(
        mode="attention"
        if budget is not None
        else "drift"
        if profile is not None
        else mode,
        profile=profile,
        budget=budget,
    )
    return hashlib.sha256(
        json.dumps(configuration, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class QualityMeasurement:
    task_id: str
    policy_id: str
    baseline_score: float
    candidate_score: float
    payload_bytes: int
    restore_ms: float
    metric: str
    provenance: str
    evidence_kind: str = "synthetic"

    def __post_init__(self):
        if any(
            not isinstance(x, str) or not x
            for x in (self.task_id, self.policy_id, self.metric, self.provenance)
        ):
            raise ValueError(
                "Quality measurements need stable identity, metric, and provenance."
            )
        if self.evidence_kind not in ("synthetic", "model_execution"):
            raise ValueError("Unknown quality evidence kind.")
        if (
            any(
                not math.isfinite(x)
                for x in (self.baseline_score, self.candidate_score, self.restore_ms)
            )
            or self.restore_ms < 0
            or type(self.payload_bytes) is not int
            or self.payload_bytes < 0
        ):
            raise ValueError("Invalid measured quality/cost values.")


@dataclass(frozen=True)
class PrecisionPolicy:
    model_key: str
    block_size: int
    config_fingerprint: str | None = None
    alpha: float = 0.5
    tau_low: float = 0.0
    tau_high: float = 1.0
    v4_error_limit: float = 0.0
    mode: str = "candidate"
    fixed_format: str | None = None
    calibration_tasks: tuple[str, ...] = ()
    validation_tasks: tuple[str, ...] = ()
    evidence: tuple[QualityMeasurement, ...] = ()
    max_quality_loss: float = 0.0
    min_baseline_score: float = 0.0
    schema_version: int = 1
    attention_fingerprint: str | None = None
    codec: str = "symmetric_token_head_le_v1"
    repair_fingerprint: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "calibration_tasks", tuple(self.calibration_tasks))
        object.__setattr__(self, "validation_tasks", tuple(self.validation_tasks))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if (
            not isinstance(self.model_key, str)
            or not self.model_key
            or type(self.block_size) is not int
            or self.block_size <= 0
            or type(self.schema_version) is not int
            or self.schema_version != 1
            or self.codec != "symmetric_token_head_le_v1"
        ):
            raise ValueError("Invalid precision-policy identity or schema.")
        if (
            any(
                not math.isfinite(x)
                for x in (
                    self.alpha,
                    self.tau_low,
                    self.tau_high,
                    self.v4_error_limit,
                    self.max_quality_loss,
                    self.min_baseline_score,
                )
            )
            or not 0 <= self.alpha <= 1
            or not 0 <= self.tau_low < self.tau_high
            or self.v4_error_limit < 0
            or self.max_quality_loss < 0
        ):
            raise ValueError("Invalid risk or quality thresholds.")
        if self.mode not in ("fixed_experiment", "candidate", "calibrated"):
            raise ValueError("Unknown precision-policy mode.")
        if self.config_fingerprint is not None and (
            len(self.config_fingerprint) != 64
            or any(x not in "0123456789abcdef" for x in self.config_fingerprint)
        ):
            raise ValueError("Invalid precision configuration fingerprint.")
        if self.attention_fingerprint is not None and (
            len(self.attention_fingerprint) != 64
            or any(x not in "0123456789abcdef" for x in self.attention_fingerprint)
        ):
            raise ValueError("Invalid attention configuration fingerprint.")
        if self.repair_fingerprint is not None and (
            len(self.repair_fingerprint) != 64
            or any(x not in "0123456789abcdef" for x in self.repair_fingerprint)
        ):
            raise ValueError("Invalid repair-policy fingerprint.")
        if self.mode != "fixed_experiment" and self.config_fingerprint is None:
            raise ValueError("Risk policies must bind the actual model configuration.")
        if (
            self.mode == "fixed_experiment"
            and self.fixed_format not in ("FP16", "K8/V8", "K8/V4")
            or self.mode != "fixed_experiment"
            and self.fixed_format is not None
        ):
            raise ValueError(
                "Only explicit fixed experiments may override risk selection."
            )
        tasks = self.calibration_tasks + self.validation_tasks
        if len(set(tasks)) != len(tasks):
            raise ValueError(
                "Calibration and validation task IDs must be unique and disjoint."
            )
        if self.mode == "calibrated" and (
            self.attention_fingerprint is None
            or len(self.calibration_tasks) < 2
            or len(self.validation_tasks) < 2
            or {x.task_id for x in self.evidence} != set(tasks)
            or len(self.evidence) != len(tasks)
            or len({x.metric for x in self.evidence}) != 1
            or any(
                x.policy_id != self.policy_id
                or x.evidence_kind != "model_execution"
                or x.baseline_score - x.candidate_score > self.max_quality_loss
                or x.baseline_score < self.min_baseline_score
                for x in self.evidence
            )
        ):
            raise ValueError(
                "Calibrated status needs passing real held-out quality evidence."
            )

    @property
    def policy_id(self):
        values = {
            key: getattr(self, key)
            for key in (
                "model_key",
                "block_size",
                "config_fingerprint",
                "alpha",
                "tau_low",
                "tau_high",
                "v4_error_limit",
                "fixed_format",
                "attention_fingerprint",
                "codec",
                "repair_fingerprint",
            )
        }
        if self.repair_fingerprint is None:
            values.pop("repair_fingerprint", None)
        return hashlib.sha256(
            json.dumps(values, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    @property
    def quality_calibrated(self):
        return self.mode == "calibrated"

    @classmethod
    def experimental_fixed(cls, model_key, block_size, format):
        return cls(model_key, block_size, mode="fixed_experiment", fixed_format=format)

    def validate_context(self, metadata):
        if (
            self.model_key != metadata.model_key
            or self.block_size != metadata.block_size
            or self.config_fingerprint is not None
            and self.config_fingerprint
            != transport_config_fingerprint(metadata.model_config_json)
        ):
            raise ValueError(
                "Precision policy does not match the checkpoint configuration."
            )

        if self.attention_fingerprint is not None and (
            metadata.attention_json is None
            or self.attention_fingerprint
            != attention_config_fingerprint(
                metadata.attention_json, len(metadata.original_ids)
            )
        ):
            raise ValueError(
                "Precision policy attention collection settings do not match."
            )

    def choose(self, risk: float, error4: float):
        if (
            not math.isfinite(risk)
            or risk < 0
            or not math.isfinite(error4)
            or error4 < 0
        ):
            raise ValueError(
                "Reconstruction risk/residual must be finite and nonnegative."
            )
        if self.mode == "fixed_experiment":
            return self.fixed_format
        if risk >= self.tau_high:
            return "FP16"
        if risk >= self.tau_low:
            return "K8/V8"
        return "K8/V4" if error4 <= self.v4_error_limit else "K8/V8"

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(self), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for name in ("calibration_tasks", "validation_tasks"):
            values[name] = tuple(values.get(name, ()))
        values["evidence"] = tuple(
            QualityMeasurement(**x) for x in values.get("evidence", ())
        )
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def calibrate_precision_policy(
    candidates,
    calibration_tasks,
    validation_tasks,
    evaluate,
    max_quality_loss,
    latency_weight_bytes_per_ms=0.0,
    min_baseline_score=0.5,
):
    """Select thresholds with whole-policy task quality, then validate once.

    evaluate(policy, task_id) must measure downstream quality of the complete
    mixed-format restore versus full precision on the identical request/repair
    set, plus actual payload and restoration cost. Per-block tensor error alone
    cannot produce calibrated status. Synthetic tests remain uncalibrated.
    """
    calibration_tasks, validation_tasks = (
        tuple(calibration_tasks),
        tuple(validation_tasks),
    )
    if (
        len(calibration_tasks) < 2
        or len(validation_tasks) < 2
        or len(set(calibration_tasks + validation_tasks))
        != len(calibration_tasks + validation_tasks)
        or not candidates
        or not math.isfinite(max_quality_loss)
        or max_quality_loss < 0
        or not math.isfinite(latency_weight_bytes_per_ms)
        or latency_weight_bytes_per_ms < 0
        or not math.isfinite(min_baseline_score)
    ):
        raise ValueError(
            "Need disjoint task splits, rules, and finite quality/cost targets."
        )
    identities = {
        (
            p.model_key,
            p.block_size,
            p.config_fingerprint,
            p.attention_fingerprint,
            p.repair_fingerprint,
        )
        for p in candidates
    }
    if len(identities) != 1 or any(p.mode != "candidate" for p in candidates):
        raise ValueError(
            "Candidates must share a model/configuration and use risk rules."
        )
    measured = []
    best = None
    baselines = {}

    def measure(policy, tasks):
        rows = [evaluate(policy, task) for task in tasks]
        if any(
            not isinstance(row, QualityMeasurement)
            or row.task_id != task
            or row.policy_id != policy.policy_id
            for row, task in zip(rows, tasks)
        ):
            raise ValueError(
                "Quality evaluator returned mismatched task/policy evidence."
            )
        for row in rows:
            baseline = (row.baseline_score, row.metric)
            if row.task_id in baselines and baselines[row.task_id] != baseline:
                raise ValueError(
                    "Baseline quality must be fixed across policy candidates."
                )
            baselines[row.task_id] = baseline
        return rows

    for policy in candidates:
        rows = measure(policy, calibration_tasks)
        measured.extend(rows)
        if all(
            row.baseline_score >= min_baseline_score
            and row.baseline_score - row.candidate_score <= max_quality_loss
            for row in rows
        ):
            cost = sum(
                row.payload_bytes + latency_weight_bytes_per_ms * row.restore_ms
                for row in rows
            )
            # Stable policy ID resolves identical measured costs deterministically.
            measured_cost = (cost, policy.policy_id, policy, rows)
            if best is None or measured_cost[:2] < best[:2]:
                best = measured_cost
    if best is None:
        raise ValueError(
            "No candidate passed measured calibration quality; use full precision."
        )
    _, _, policy, rows = best
    validation = measure(policy, validation_tasks)
    if any(
        row.baseline_score < min_baseline_score
        or row.baseline_score - row.candidate_score > max_quality_loss
        for row in validation
    ):
        raise ValueError("Selected precision rule failed held-out generation quality.")
    evidence = tuple(rows + validation)
    if len({row.metric for row in measured + validation}) != 1:
        raise ValueError("Calibration quality metrics must be consistent.")
    real = policy.attention_fingerprint is not None and all(
        row.evidence_kind == "model_execution" for row in evidence
    )
    result = replace(
        policy,
        mode="calibrated" if real else "candidate",
        calibration_tasks=calibration_tasks,
        validation_tasks=validation_tasks,
        evidence=evidence,
        max_quality_loss=max_quality_loss,
        min_baseline_score=min_baseline_score,
    )
    return result, tuple(measured + validation)
