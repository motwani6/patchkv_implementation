# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-time attention scoring and deterministic PatchKV selection."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def token_digest(token_ids: Sequence[int]) -> str:
    if any(type(x) is not int or x < 0 for x in token_ids):
        raise ValueError("Token IDs must be nonnegative integers.")
    return hashlib.sha256(
        json.dumps(list(token_ids), separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class AttentionLayerSpec:
    """Actual attention geometry and post-RoPE identity for one sampled layer."""

    name: str
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    scale: float
    rope_signature: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.rope_signature, str)
            or not self.rope_signature
        ):
            raise ValueError("Layer name and RoPE signature are required.")
        for name in ("num_query_heads", "num_kv_heads", "head_size"):
            _positive(getattr(self, name), name)
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads.")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Attention scale must be finite and positive.")


@dataclass(frozen=True)
class AttentionBudget:
    """A token count or fraction of eligible suffix tokens, before rounding."""

    max_tokens: int | None = None
    fraction: float | None = None

    def __post_init__(self) -> None:
        if (self.max_tokens is None) == (self.fraction is None):
            raise ValueError("Choose exactly one token-count or fractional budget.")
        if self.max_tokens is not None and (
            type(self.max_tokens) is not int or self.max_tokens < 0
        ):
            raise ValueError("Token budget must be a nonnegative integer.")
        if self.fraction is not None and (
            isinstance(self.fraction, bool)
            or not math.isfinite(self.fraction)
            or not 0 <= self.fraction <= 1
        ):
            raise ValueError("Fractional budget must be finite and in [0, 1].")

    def count(self, eligible: int) -> int:
        if type(eligible) is not int or eligible < 0:
            raise ValueError("Eligible count must be a nonnegative integer.")
        return min(
            eligible,
            self.max_tokens
            if self.max_tokens is not None
            else math.ceil(self.fraction * eligible),
        )


@dataclass(frozen=True)
class AttentionMetadata:
    """Normalized old-context scores; data-only checkpoint metadata, not KV."""

    model_key: str
    checkpoint_id: str
    token_digest: str
    token_count: int
    block_size: int
    query_positions: tuple[int, ...]
    layers: tuple[AttentionLayerSpec, ...]
    scores: tuple[float, ...]
    schema_version: int = 1
    scoring: str = "post_rope_fp32_causal_gqa_layer_mean_l1"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported attention metadata schema.")
        if self.scoring != "post_rope_fp32_causal_gqa_layer_mean_l1":
            raise ValueError("Unsupported attention scoring method.")
        if any(
            not isinstance(x, str) or not x
            for x in (self.model_key, self.checkpoint_id)
        ):
            raise ValueError(
                "Model/configuration and checkpoint identities are required."
            )
        if (
            not isinstance(self.token_digest, str)
            or len(self.token_digest) != 64
            or any(x not in "0123456789abcdef" for x in self.token_digest)
        ):
            raise ValueError("Invalid token digest.")
        _positive(self.token_count, "token_count")
        _positive(self.block_size, "block_size")
        if (
            not self.query_positions
            or any(type(x) is not int or x < 0 for x in self.query_positions)
            or tuple(sorted(set(self.query_positions))) != self.query_positions
        ):
            raise ValueError(
                "Query positions must be sorted, unique nonnegative integers."
            )
        if not self.layers or len({x.name for x in self.layers}) != len(self.layers):
            raise ValueError("Sampled layers must be nonempty and unique.")
        if (
            len(self.scores) != self.token_count
            or any(not math.isfinite(x) or x < 0 for x in self.scores)
            or not math.isclose(math.fsum(self.scores), 1.0, abs_tol=1e-6)
        ):
            raise ValueError(
                "Expected one finite, nonnegative L1-normalized score per token."
            )

    def validate_context(
        self,
        original_ids: Sequence[int],
        *,
        model_key: str,
        block_size: int,
        checkpoint_id: str,
        layers: Sequence[AttentionLayerSpec] | None = None,
        query_positions: Sequence[int] | None = None,
    ) -> None:
        if (
            model_key != self.model_key
            or block_size != self.block_size
            or checkpoint_id != self.checkpoint_id
            or len(original_ids) != self.token_count
            or token_digest(original_ids) != self.token_digest
            or (layers is not None and tuple(layers) != self.layers)
            or (
                query_positions is not None
                and tuple(query_positions) != self.query_positions
            )
        ):
            raise ValueError(
                "Attention metadata does not match the source checkpoint/configuration."
            )

    def select_suffix_offsets(
        self,
        old_suffix_start: int,
        dirty_tokens: int,
        budget: AttentionBudget,
    ) -> tuple[int, ...]:
        """Rank only S minus D, with lower original position breaking ties."""
        if (
            type(old_suffix_start) is not int
            or not 0 <= old_suffix_start <= self.token_count
            or type(dirty_tokens) is not int
            or not 0 <= dirty_tokens <= self.token_count - old_suffix_start
        ):
            raise ValueError("Invalid suffix boundary or dirty-window length.")
        first = old_suffix_start + dirty_tokens
        count = budget.count(self.token_count - first)
        selected = heapq.nsmallest(
            count, range(first, self.token_count), key=lambda i: (-self.scores[i], i)
        )
        return tuple(sorted(i - old_suffix_start for i in selected))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(self), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> AttentionMetadata:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, values: dict) -> AttentionMetadata:
        data = dict(values)
        data["layers"] = tuple(AttentionLayerSpec(**x) for x in data["layers"])
        data["scores"] = tuple(data["scores"])
        data["query_positions"] = tuple(data["query_positions"])
        return cls(**data)


def streaming_attention_mass(
    query: Any,
    keys: Any,
    query_positions: Sequence[int],
    *,
    scale: float,
    key_tile: int = 256,
    query_tile: int = 2,
    head_tile: int = 8,
) -> Any:
    """Return mean attention over Q/H using bounded FP32 two-pass softmax tiles.

    Inputs are already RoPE-positioned [Q, Hq, D] and [K, Hkv, D] tensors.
    Keys occupy contiguous absolute positions [0, K). Probe/generated keys must
    be included in the denominator even if not part of the checkpoint payload.
    No [Q, Hq, K] attention tensor is allocated; temporary logits are bounded by
    query_tile * head_tile * key_tile. Each GQA group maps consecutive Q heads
    to one KV head. All keys after each query's absolute position are masked.
    """
    import torch

    for name, value in (
        ("key_tile", key_tile),
        ("query_tile", query_tile),
        ("head_tile", head_tile),
    ):
        _positive(value, name)
    if (
        not isinstance(query, torch.Tensor)
        or not isinstance(keys, torch.Tensor)
        or query.ndim != 3
        or keys.ndim != 3
        or query.device != keys.device
        or not query.is_floating_point()
        or not keys.is_floating_point()
        or not all(query.shape)
        or not all(keys.shape)
        or query.shape[2] != keys.shape[2]
        or query.shape[1] % keys.shape[1]
        or not math.isfinite(scale)
        or scale <= 0
        or query.requires_grad
        or keys.requires_grad
    ):
        raise ValueError(
            "Expected detached compatible Q/K tensors and a positive scale."
        )
    if (
        len(query_positions) != len(query)
        or any(type(x) is not int or not 0 <= x < len(keys) for x in query_positions)
        or len(set(query_positions)) != len(query_positions)
    ):
        raise ValueError(
            "Every query needs a unique absolute position with a cached key."
        )
    # Validate in tiles, without allocating a second full-context K tensor.
    for tensor in (query, keys):
        for chunk in tensor.split(key_tile):
            if not torch.isfinite(chunk).all():
                raise ValueError("Q/K must be finite.")
    with torch.inference_mode():
        mass = torch.zeros(len(keys), dtype=torch.float32, device=query.device)
        group = query.shape[1] // keys.shape[1]
        for qlo in range(0, len(query), query_tile):
            positions = torch.tensor(
                query_positions[qlo : qlo + query_tile], device=query.device
            )
            for hlo in range(0, query.shape[1], head_tile):
                q = query[qlo : qlo + query_tile, hlo : hlo + head_tile].float()
                mapping = (
                    torch.arange(hlo, hlo + q.shape[1], device=query.device) // group
                )
                maximum = torch.full(q.shape[:2], -torch.inf, device=query.device)
                denominator = torch.zeros_like(maximum)

                def logits(klo, q=q, mapping=mapping, positions=positions):
                    k = keys[klo : klo + key_tile, mapping].float()
                    scores = torch.einsum("qhd,khd->qhk", q, k) * scale
                    future = (
                        torch.arange(klo, klo + len(k), device=q.device)[None, :]
                        > positions[:, None]
                    )
                    scores.masked_fill_(future[:, None, :], -torch.inf)
                    return scores

                for klo in range(0, len(keys), key_tile):
                    scores = logits(klo)
                    new_max = torch.maximum(maximum, scores.amax(-1))
                    safe_max = torch.where(torch.isfinite(new_max), new_max, 0)
                    denominator = denominator * torch.exp(
                        maximum - safe_max
                    ) + torch.exp(scores - safe_max[..., None]).sum(-1)
                    maximum = new_max
                if not torch.isfinite(denominator).all() or not (denominator > 0).all():
                    raise ValueError(
                        "Nonfinite attention logits or empty causal context."
                    )
                for klo in range(0, len(keys), key_tile):
                    scores = logits(klo)
                    weights = (
                        torch.exp(scores - maximum[..., None]) / denominator[..., None]
                    )
                    mass[klo : klo + scores.shape[-1]] += weights.sum((0, 1))
        return mass / (len(query) * query.shape[1])


class AttentionAccumulator:
    """Collect selected post-RoPE Q and all causally needed K on CPU."""

    def __init__(
        self,
        original_ids: Sequence[int],
        *,
        model_key: str,
        checkpoint_id: str,
        block_size: int,
        query_positions: Sequence[int],
        layers: Sequence[AttentionLayerSpec],
        key_tile: int = 256,
        query_tile: int = 2,
        head_tile: int = 8,
    ) -> None:
        # Validate all static metadata before installing hooks or allocating Q/K.
        self.template = AttentionMetadata(
            model_key,
            checkpoint_id,
            token_digest(original_ids),
            len(original_ids),
            block_size,
            tuple(query_positions),
            tuple(layers),
            tuple([1.0 / len(original_ids)] * len(original_ids))
            if original_ids
            else (),
        )
        for name, value in (
            ("key_tile", key_tile),
            ("query_tile", query_tile),
            ("head_tile", head_tile),
        ):
            _positive(value, name)
        self.tiles = dict(key_tile=key_tile, query_tile=query_tile, head_tile=head_tile)
        self.limit = max(query_positions) + 1
        self.keys: dict[str, Any] = {}
        self.seen: dict[str, set[int]] = {x.name: set() for x in layers}
        self.queries: dict[str, dict[int, Any]] = {x.name: {} for x in layers}
        self.specs = {x.name: x for x in layers}

    def record(self, layer: str, positions: Any, query: Any, keys: Any) -> None:
        """Capture a forward chunk; fail closed on repeated/missing source rows."""
        import torch

        if layer not in self.specs:
            raise ValueError("Layer is not part of the configured sample.")
        spec = self.specs[layer]
        if (
            positions.ndim != 1
            or positions.dtype not in (torch.int32, torch.int64)
            or query.shape != (len(positions), spec.num_query_heads, spec.head_size)
            or keys.shape != (len(positions), spec.num_kv_heads, spec.head_size)
            or not query.is_floating_point()
            or not keys.is_floating_point()
        ):
            raise ValueError("Invalid positions or post-RoPE Q/K shapes.")
        absolute = positions.detach().cpu().tolist()
        if any(x < 0 for x in absolute) or absolute != sorted(set(absolute)):
            raise ValueError(
                "Chunk positions must be unique, ordered, and nonnegative."
            )
        rows = [i for i, pos in enumerate(absolute) if pos < self.limit]
        selected_positions = [absolute[i] for i in rows]
        if self.seen[layer].intersection(selected_positions):
            raise ValueError(
                "Duplicate source positions; use one uncached checkpoint pass."
            )
        if not rows:
            return
        if layer not in self.keys:
            self.keys[layer] = torch.empty(
                self.limit,
                spec.num_kv_heads,
                spec.head_size,
                dtype=keys.dtype,
                device="cpu",
            )
        elif self.keys[layer].dtype != keys.dtype:
            raise ValueError("KV dtype changed during attention collection.")
        source_rows = torch.tensor(rows, device=keys.device)
        self.keys[layer][selected_positions] = keys[source_rows].detach().cpu()
        for row, position in zip(rows, selected_positions):
            if position in self.template.query_positions:
                self.queries[layer][position] = query[row].detach().cpu().clone()
        self.seen[layer].update(selected_positions)

    def finalize(self) -> AttentionMetadata:
        import torch

        total = torch.zeros(self.template.token_count, dtype=torch.float64)
        for spec in self.template.layers:
            if len(self.seen[spec.name]) != self.limit or set(
                self.queries[spec.name]
            ) != set(self.template.query_positions):
                raise ValueError("Missing keys/queries in the checkpoint pass.")
            query = torch.stack(
                [self.queries[spec.name][pos] for pos in self.template.query_positions]
            )
            mass = streaming_attention_mass(
                query,
                self.keys[spec.name],
                self.template.query_positions,
                scale=spec.scale,
                **self.tiles,
            )
            count = min(len(mass), len(total))
            total[:count] += mass[:count].double()
        if not torch.isfinite(total).all() or total.sum() <= 0:
            raise ValueError("No finite attention mass on checkpoint tokens.")
        from dataclasses import replace

        return replace(self.template, scores=tuple((total / total.sum()).tolist()))
