# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline value-drift calibration and dependency-free length prediction."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _lengths(*values: int) -> None:
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Lengths must be nonnegative integers.")


def _fraction(value: float, name: str, *, upper: float = 1.0) -> None:
    if not math.isfinite(value) or not 0 < value <= upper:
        raise ValueError(f"{name} must be finite and in (0, {upper}].")


def _quantile(values: Sequence[float], q: float) -> float:
    """Return an upper empirical quantile without interpolation."""
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def _write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@dataclass(frozen=True)
class DriftCalibration:
    """Model-specific settings for equations (6) and (7)."""

    model_key: str
    layers: tuple[str, ...]
    block_size: int
    deviation_threshold: float
    stability_coverage: float
    aggregate_quantile: float = 0.9

    def __post_init__(self) -> None:
        if not isinstance(self.model_key, str) or not self.model_key:
            raise ValueError("A model/revision/configuration identity is required.")
        if not self.layers or any(not isinstance(x, str) or not x for x in self.layers):
            raise ValueError("At least one named profiling layer is required.")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("Profiling layers must be unique.")
        _lengths(self.block_size)
        if not self.block_size:
            raise ValueError("Block size must be positive.")
        if not math.isfinite(self.deviation_threshold) or not (
            0 <= self.deviation_threshold <= 2
        ):
            raise ValueError("Cosine deviation threshold must be in [0, 2].")
        _fraction(self.stability_coverage, "stability_coverage")
        _fraction(self.aggregate_quantile, "aggregate_quantile")


@dataclass(frozen=True)
class ProfilePair:
    """One task-grouped, single-span edit in token coordinates."""

    pair_id: str
    task_id: str
    original_ids: tuple[int, ...]
    start: int
    end: int
    replacement_ids: tuple[int, ...]

    @classmethod
    def from_contexts(
        cls,
        pair_id: str,
        task_id: str,
        original_ids: Sequence[int],
        edited_ids: Sequence[int],
    ) -> ProfilePair:
        """Find the longest prefix and maximal nonoverlapping aligned suffix."""
        old, new = tuple(original_ids), tuple(edited_ids)
        prefix = 0
        while prefix < min(len(old), len(new)) and old[prefix] == new[prefix]:
            prefix += 1
        suffix = 0
        while (
            suffix < min(len(old), len(new)) - prefix
            and old[len(old) - suffix - 1] == new[len(new) - suffix - 1]
        ):
            suffix += 1
        return cls(
            pair_id,
            task_id,
            old,
            prefix,
            len(old) - suffix,
            new[prefix : len(new) - suffix],
        )

    def __post_init__(self) -> None:
        if any(not isinstance(x, str) or not x for x in (self.pair_id, self.task_id)):
            raise ValueError("Stable pair and task IDs are required.")
        _lengths(self.start, self.end)
        if not self.start <= self.end <= len(self.original_ids):
            raise ValueError("Expected start <= end <= original length.")
        if any(
            type(x) is not int or x < 0
            for x in (*self.original_ids, *self.replacement_ids)
        ):
            raise ValueError("Token IDs must be nonnegative integers.")

    @property
    def edited_ids(self) -> tuple[int, ...]:
        return (
            self.original_ids[: self.start]
            + self.replacement_ids
            + self.original_ids[self.end :]
        )


def oracle_dirty_window(
    block_deviations: Sequence[float], suffix_length: int, config: DriftCalibration
) -> int:
    """Implement equation (7), considering nonempty suffix-relative block tails."""
    _lengths(suffix_length)
    if len(block_deviations) != math.ceil(suffix_length / config.block_size):
        raise ValueError("Deviation count must match suffix-relative blocks.")
    if any(not math.isfinite(x) or not 0 <= x <= 2 for x in block_deviations):
        raise ValueError("Block deviations must be finite and in [0, 2].")
    stable = 0
    boundary = suffix_length
    for index in range(len(block_deviations) - 1, -1, -1):
        stable += block_deviations[index] <= config.deviation_threshold
        if stable / (len(block_deviations) - index) >= config.stability_coverage:
            boundary = index * config.block_size
    return boundary


@dataclass(frozen=True)
class DriftObservation:
    """Measured block deviations for one exact paired-prefill sample."""

    pair_id: str
    task_id: str
    prefix_length: int
    replacement_length: int
    suffix_length: int
    block_deviations: tuple[float, ...]
    config: DriftCalibration

    def __post_init__(self) -> None:
        if any(not isinstance(x, str) or not x for x in (self.pair_id, self.task_id)):
            raise ValueError("Pair and task IDs are required.")
        _lengths(*self.lengths)
        oracle_dirty_window(self.block_deviations, self.suffix_length, self.config)

    @property
    def lengths(self) -> tuple[int, int, int]:
        return self.prefix_length, self.replacement_length, self.suffix_length

    @property
    def dirty_window(self) -> int:
        return oracle_dirty_window(
            self.block_deviations, self.suffix_length, self.config
        )


def observe_values(
    pair: ProfilePair,
    old_values: Mapping[str, Any],
    new_values: Mapping[str, Any],
    config: DriftCalibration,
) -> DriftObservation:
    """Measure equation (6) on CPU tensors shaped [tokens, heads, head_dim].

    Heads are flattened for each token's cosine; block aggregation pools sampled
    layers and tokens using an upper empirical quantile. Zero/zero has deviation
    zero; zero/nonzero has deviation one. Tensors must be detached CPU values
    from exact prefills, never keys or approximate edit output.
    """
    import torch

    suffix_length = len(pair.original_ids) - pair.end
    suffix_start = pair.start + len(pair.replacement_ids)
    deviations = []
    for layer in config.layers:
        if layer not in old_values or layer not in new_values:
            raise ValueError(f"Missing profiling layer: {layer}.")
        old, new = old_values[layer], new_values[layer]
        for tensor, length in (
            (old, len(pair.original_ids)),
            (new, len(pair.edited_ids)),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.requires_grad
                or tensor.ndim != 3
                or tensor.shape[0] != length
                or not tensor.is_floating_point()
                or not all(tensor.shape[1:])
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(
                    "Expected finite detached CPU [tokens, heads, dim] values."
                )
        if old.shape[1:] != new.shape[1:]:
            raise ValueError("Paired value head shapes must match.")
        a = old[pair.end :].double().flatten(1)
        b = new[suffix_start:].double().flatten(1)
        norm = torch.linalg.vector_norm(a, dim=1) * torch.linalg.vector_norm(b, dim=1)
        cosine = torch.where(norm > 0, (a * b).sum(1) / norm.clamp_min(1e-300), 0)
        both_zero = (a == 0).all(1) & (b == 0).all(1)
        deviations.append(torch.where(both_zero, 0, 1 - cosine.clamp(-1, 1)))
    blocks = []
    for start in range(0, suffix_length, config.block_size):
        pooled = torch.cat([d[start : start + config.block_size] for d in deviations])
        blocks.append(_quantile(pooled.tolist(), config.aggregate_quantile))
    return DriftObservation(
        pair.pair_id,
        pair.task_id,
        pair.start,
        len(pair.replacement_ids),
        suffix_length,
        tuple(blocks),
        config,
    )


def profile_pair(
    pair: ProfilePair,
    exact_prefill: Callable[[tuple[int, ...]], Mapping[str, Any]],
    config: DriftCalibration,
) -> DriftObservation:
    """Run two independent exact prefills through a caller-provided adapter.

    The adapter must return detached CPU value tensors for every sampled layer
    and disable cache reuse. Copy old output before the second call so an adapter
    can safely reuse its output buffers. GPU execution belongs in the adapter.
    """
    old = {
        name: tensor.clone()
        for name, tensor in exact_prefill(pair.original_ids).items()
    }
    new = exact_prefill(pair.edited_ids)
    return observe_values(pair, old, new, config)


def split_by_task(
    samples: Sequence[DriftObservation],
    calibration_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[list[DriftObservation], list[DriftObservation]]:
    """Split whole tasks, keeping adjacent revisions in one partition."""
    _fraction(calibration_fraction, "calibration_fraction")
    if calibration_fraction == 1:
        raise ValueError("Training partition must be nonempty.")
    groups = sorted({s.task_id for s in samples})
    if len(groups) < 2:
        raise ValueError("At least two independent tasks are required.")
    random.Random(seed).shuffle(groups)
    count = max(1, min(len(groups) - 1, math.ceil(len(groups) * calibration_fraction)))
    calibration = set(groups[:count])
    return (
        [s for s in samples if s.task_id not in calibration],
        [s for s in samples if s.task_id in calibration],
    )


def _features(lengths: Sequence[int]) -> tuple[float, ...]:
    _lengths(*lengths)
    p, m, s = (math.log1p(x) for x in lengths)
    return 1.0, p, m, s, p * m, p * s, m * s


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _least_squares_start(
    rows: Sequence[Sequence[float]], targets: Sequence[float]
) -> list[float]:
    matrix = [
        [
            sum(row[i] * row[j] for row in rows) + (1e-8 if i == j else 0)
            for j in range(7)
        ]
        + [sum(row[i] * y for row, y in zip(rows, targets))]
        for i in range(7)
    ]
    for i in range(7):
        pivot = max(range(i, 7), key=lambda j: abs(matrix[j][i]))
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        divisor = matrix[i][i]
        matrix[i] = [x / divisor for x in matrix[i]]
        for j in range(7):
            if j != i:
                factor = matrix[j][i]
                matrix[j] = [a - factor * b for a, b in zip(matrix[j], matrix[i])]
    return [row[-1] for row in matrix]


@dataclass(frozen=True)
class DriftProfile:
    """Frozen high-quantile length model plus task-held-out safety calibration."""

    config: DriftCalibration
    quantile: float
    coefficients: tuple[float, ...]
    feature_scales: tuple[float, ...]
    residual_margin: float | None
    length_bounds: tuple[tuple[int, int], ...]
    training_tasks: tuple[str, ...]
    calibration_tasks: tuple[str, ...]
    training_pairs: int
    calibration_pairs: int
    training_pinball_loss: float
    fit_sweeps: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        _fraction(self.quantile, "quantile")
        if self.quantile <= 0.5 or self.quantile == 1:
            raise ValueError("Predictor quantile must be in (0.5, 1).")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported drift-profile schema.")
        if (
            len(self.coefficients) != 7
            or len(self.feature_scales) != 7
            or any(not math.isfinite(x) for x in self.coefficients)
            or any(not math.isfinite(x) or x <= 0 for x in self.feature_scales)
        ):
            raise ValueError("Expected seven finite coefficients and positive scales.")
        if self.residual_margin is not None and (
            not math.isfinite(self.residual_margin) or self.residual_margin < 0
        ):
            raise ValueError("Residual safety margin must be nonnegative or null.")
        if len(self.length_bounds) != 3:
            raise ValueError("Expected three calibrated length bounds.")
        for lo, hi in self.length_bounds:
            _lengths(lo, hi)
            if lo > hi:
                raise ValueError("Invalid calibrated length interval.")
        if (
            not self.training_tasks
            or not self.calibration_tasks
            or set(self.training_tasks) & set(self.calibration_tasks)
        ):
            raise ValueError(
                "Training and calibration tasks must be nonempty and disjoint."
            )
        if len(set(self.training_tasks)) != len(self.training_tasks) or len(
            set(self.calibration_tasks)
        ) != len(self.calibration_tasks):
            raise ValueError("Task IDs must be unique within each split.")
        _lengths(self.training_pairs, self.calibration_pairs, self.fit_sweeps)
        if (
            self.training_pairs < len(self.training_tasks)
            or self.calibration_pairs < len(self.calibration_tasks)
            or not self.fit_sweeps
            or not math.isfinite(self.training_pinball_loss)
            or self.training_pinball_loss < 0
        ):
            raise ValueError("Invalid fitting metadata.")
        if (
            math.ceil((len(self.calibration_tasks) + 1) * self.quantile)
            > len(self.calibration_tasks)
            and self.residual_margin is not None
        ):
            raise ValueError("Insufficient calibration tasks for a finite margin.")

    def _raw(self, lengths: Sequence[int]) -> float:
        return _dot(
            self.coefficients,
            [x / s for x, s in zip(_features(lengths), self.feature_scales)],
        )

    def predict(
        self, prefix_length: int, replacement_length: int, suffix_length: int
    ) -> int:
        """Predict tokens, falling back to full repair outside calibrated support."""
        lengths = prefix_length, replacement_length, suffix_length
        _lengths(*lengths)
        if self.residual_margin is None or any(
            not lo <= value <= hi
            for value, (lo, hi) in zip(lengths, self.length_bounds)
        ):
            return suffix_length
        return min(
            suffix_length, max(0, math.ceil(self._raw(lengths) + self.residual_margin))
        )

    def save(self, path: str | Path) -> None:
        """Write a versioned, data-only JSON artifact; no pickle or tensors."""
        _write_json(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> DriftProfile:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["config"] = DriftCalibration(
            **{**data["config"], "layers": tuple(data["config"]["layers"])}
        )
        for key in (
            "coefficients",
            "feature_scales",
            "training_tasks",
            "calibration_tasks",
        ):
            data[key] = tuple(data[key])
        data["length_bounds"] = tuple(tuple(x) for x in data["length_bounds"])
        return cls(**data)


def fit_drift_profile(
    training: Sequence[DriftObservation],
    calibration: Sequence[DriftObservation],
    quantile: float = 0.95,
    max_sweeps: int = 200,
) -> DriftProfile:
    """Fit pinball loss by exact coordinate updates, then calibrate by task.

    Features are log1p lengths and pairwise interactions. A held-out upper
    quantile of each task's maximum underprediction is added as a safety margin.
    Too few calibration tasks yields full repair, not an unsupported guarantee.
    Coordinate descent is deterministic but not a global-optimality certificate.
    """
    _fraction(quantile, "quantile")
    _lengths(max_sweeps)
    if not 0.5 < quantile < 1 or not max_sweeps:
        raise ValueError("Need quantile in (0.5, 1) and positive max_sweeps.")
    if not training or not calibration:
        raise ValueError("Both training and independent calibration data are required.")
    config = training[0].config
    all_samples = [*training, *calibration]
    if any(s.config != config for s in all_samples):
        raise ValueError("Cannot mix model identities or drift calibrations.")
    if len({s.pair_id for s in all_samples}) != len(all_samples):
        raise ValueError("Pair IDs must be unique across both partitions.")
    train_tasks = tuple(sorted({s.task_id for s in training}))
    cal_tasks = tuple(sorted({s.task_id for s in calibration}))
    if set(train_tasks) & set(cal_tasks):
        raise ValueError("Task leakage between training and calibration.")
    features = [_features(s.lengths) for s in training]
    scales = tuple(max(1.0, max(abs(row[j]) for row in features)) for j in range(7))
    rows = [[x / scale for x, scale in zip(row, scales)] for row in features]
    targets = [float(s.dirty_window) for s in training]
    coefficients = _least_squares_start(rows, targets)
    coefficients[0] += _quantile(
        [y - _dot(row, coefficients) for row, y in zip(rows, targets)], quantile
    )

    def loss() -> float:
        residuals = [y - _dot(row, coefficients) for row, y in zip(rows, targets)]
        return sum(max(quantile * r, (quantile - 1) * r) for r in residuals) / len(rows)

    previous = loss()
    for sweep in range(1, max_sweeps + 1):
        for j in range(7):
            candidates = []
            cutoff = 0.0
            for row, target in zip(rows, targets):
                x = row[j]
                if abs(x) < 1e-15:
                    continue
                residual = target - _dot(row, coefficients) + x * coefficients[j]
                candidates.append((residual / x, abs(x)))
                cutoff += abs(x) * (quantile if x > 0 else 1 - quantile)
            if candidates:
                cumulative = 0.0
                for candidate, weight in sorted(candidates):
                    cumulative += weight
                    if cumulative + 1e-12 >= cutoff:
                        coefficients[j] = candidate
                        break
        current = loss()
        if abs(previous - current) <= 1e-9 * max(1.0, previous):
            break
        previous = current
    group_residuals = []
    for task in cal_tasks:
        group_residuals.append(
            max(
                s.dirty_window
                - _dot(
                    coefficients,
                    [x / scale for x, scale in zip(_features(s.lengths), scales)],
                )
                for s in calibration
                if s.task_id == task
            )
        )
    rank = math.ceil((len(cal_tasks) + 1) * quantile)
    margin = (
        max(0.0, sorted(group_residuals)[rank - 1]) if rank <= len(cal_tasks) else None
    )
    bounds = tuple(
        (min(s.lengths[j] for s in all_samples), max(s.lengths[j] for s in all_samples))
        for j in range(3)
    )
    return DriftProfile(
        config,
        quantile,
        tuple(coefficients),
        scales,
        margin,
        bounds,
        train_tasks,
        cal_tasks,
        len(training),
        len(calibration),
        loss(),
        sweep,
    )


def save_observations(path: str | Path, samples: Sequence[DriftObservation]) -> None:
    _write_json(
        path, {"schema_version": 1, "observations": [asdict(s) for s in samples]}
    )


def load_observations(path: str | Path) -> list[DriftObservation]:
    result = []
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        type(artifact.get("schema_version")) is not int
        or artifact["schema_version"] != 1
    ):
        raise ValueError("Unsupported drift-observation schema.")
    for data in artifact["observations"]:
        config = data.pop("config")
        config["layers"] = tuple(config["layers"])
        data["block_deviations"] = tuple(data["block_deviations"])
        result.append(DriftObservation(**data, config=DriftCalibration(**config)))
    return result
