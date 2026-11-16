# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, full-precision CPU checkpoints independent of physical KV pages."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from threading import RLock
from typing import Any


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass(frozen=True)
class CheckpointLayer:
    name: str
    num_kv_heads: int
    head_size: int
    dtype: str
    rope_signature: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(x, str) or not x for x in (self.name, self.rope_signature)
        ):
            raise ValueError("Layer name and RoPE identity are required.")
        _integer(self.num_kv_heads, "num_kv_heads", 1)
        _integer(self.head_size, "head_size", 1)
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError(
                "Checkpoint storage requires unquantized floating-point KV."
            )

    @property
    def row_bytes(self) -> int:
        return (
            self.num_kv_heads
            * 2
            * self.head_size
            * (4 if self.dtype == "float32" else 2)
        )


@dataclass(frozen=True)
class CheckpointBlock:
    logical_index: int
    source_start: int
    count: int

    def __post_init__(self) -> None:
        _integer(self.logical_index, "logical_index")
        _integer(self.source_start, "source_start")
        _integer(self.count, "count", 1)


@dataclass(frozen=True)
class CheckpointMetadata:
    checkpoint_id: str
    model_key: str
    model_config_json: str
    original_ids: tuple[int, ...]
    source_cache_salt: str | None
    block_size: int
    layers: tuple[CheckpointLayer, ...]
    blocks: tuple[CheckpointBlock, ...]
    attention_json: str | None = None
    schema_version: int = 1
    transport_json: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_ids", tuple(self.original_ids))
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported checkpoint schema.")
        if any(
            not isinstance(x, str) or not x
            for x in (self.checkpoint_id, self.model_key)
        ):
            raise ValueError("Checkpoint and model identities are required.")
        if not isinstance(json.loads(self.model_config_json), dict):
            raise ValueError("Model configuration must be a JSON object.")
        if self.source_cache_salt is not None and not isinstance(
            self.source_cache_salt, str
        ):
            raise ValueError("Source cache salt must be a string or None.")
        _integer(self.block_size, "block_size", 1)
        if any(type(x) is not int or x < 0 for x in self.original_ids):
            raise ValueError("Original token IDs must be nonnegative integers.")
        if not self.layers or len({x.name for x in self.layers}) != len(self.layers):
            raise ValueError("Checkpoint layers must be nonempty and unique.")
        expected = tuple(
            CheckpointBlock(
                i // self.block_size,
                i,
                min(self.block_size, len(self.original_ids) - i),
            )
            for i in range(0, len(self.original_ids), self.block_size)
        )
        if self.blocks != expected:
            raise ValueError(
                "Logical block/source-position mapping is incomplete or invalid."
            )
        if self.attention_json is not None and not isinstance(
            json.loads(self.attention_json), dict
        ):
            raise ValueError("Attention metadata must be a JSON object.")
        if self.transport_json is not None:
            transport = json.loads(self.transport_json)
            if (
                transport.get("codec") != "symmetric_token_head_le_v1"
                or len(transport.get("precision_tags", [])) != len(self.blocks)
                or any(
                    tag not in ("FP16", "K8/V8", "K8/V4")
                    for tag in transport["precision_tags"]
                )
                or type(transport.get("payload_bytes")) is not int
                or transport["payload_bytes"] < 0
            ):
                raise ValueError("Invalid frozen checkpoint transport metadata.")

    @property
    def payload_bytes(self) -> int:
        if self.transport_json is not None:
            return json.loads(self.transport_json)["payload_bytes"]
        return self.source_precision_bytes

    @property
    def source_precision_bytes(self) -> int:
        return len(self.original_ids) * sum(layer.row_bytes for layer in self.layers)

    def validate(
        self,
        *,
        model_key: str,
        model_config_json: str,
        layers: tuple[CheckpointLayer, ...],
        original_ids: tuple[int, ...],
        block_size: int,
    ) -> None:
        if (
            model_key != self.model_key
            or json.loads(model_config_json) != json.loads(self.model_config_json)
            or layers != self.layers
            or original_ids != self.original_ids
            or block_size != self.block_size
        ):
            raise ValueError(
                "Checkpoint model, geometry, or original tokens do not match."
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(self), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> CheckpointMetadata:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["original_ids"] = tuple(data["original_ids"])
        data["layers"] = tuple(CheckpointLayer(**x) for x in data["layers"])
        data["blocks"] = tuple(CheckpointBlock(**x) for x in data["blocks"])
        return cls(**data)


def _deep_size(value: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if is_dataclass(value):
        size += sys.getsizeof(vars(value))
        size += sum(
            _deep_size(getattr(value, field.name), seen) for field in fields(value)
        )
    elif isinstance(value, (tuple, list)):
        size += sum(_deep_size(x, seen) for x in value)
    return size


class _CheckpointReader:
    def __init__(self, metadata, payload, transfer_tokens):
        self.metadata = metadata
        self._payload = payload
        self.transfer_tokens = transfer_tokens
        self.active = True
        self.transfer_bytes = 0
        self.format_bytes = {}

    def read(self, layer: str, start: int, count: int):
        if not self.active:
            raise ValueError("Checkpoint lease has ended.")
        _integer(start, "start")
        _integer(count, "count")
        if count > self.transfer_tokens or start + count > len(
            self.metadata.original_ids
        ):
            raise ValueError("Read exceeds checkpoint bounds or transfer chunk limit.")
        # Never expose mutable views of the store's owned storage.
        return self._payload[layer][start : start + count].clone()

    def read_to(self, layer, start, count, device):
        rows = self.read(layer, start, count)
        result = rows.to(device)
        size = rows.numel() * rows.element_size()
        self.transfer_bytes += size
        self.format_bytes["SOURCE"] = self.format_bytes.get("SOURCE", 0) + size
        return result


class CPUCheckpointStore:
    """Memory-only store with reservation, leases, explicit release, and no LRU."""

    def __init__(self, max_bytes: int, transfer_tokens: int = 256):
        _integer(max_bytes, "max_bytes", 1)
        _integer(transfer_tokens, "transfer_tokens", 1)
        self.max_bytes = max_bytes
        self.transfer_tokens = transfer_tokens
        self.used_bytes = 0
        self._entries: dict[str, Any] = {}
        self._leases: dict[str, int] = {}
        self._lock = RLock()

    @staticmethod
    def charged_bytes(metadata: CheckpointMetadata) -> int:
        # Include Python metadata and conservative tensor/dictionary allowances.
        return (
            metadata.payload_bytes
            + _deep_size(metadata)
            + 4096 * (1 + len(metadata.layers))
        )

    def create(
        self, metadata: CheckpointMetadata, read_source: Callable[[str, int, int], Any]
    ) -> None:
        import torch

        with self._lock:
            key = metadata.checkpoint_id
            if key in self._entries:
                raise ValueError("Checkpoint identity already exists.")
            charge = self.charged_bytes(metadata)
            if self.used_bytes + charge > self.max_bytes:
                raise MemoryError("CPU checkpoint memory budget exhausted.")
            self.used_bytes += charge
            payload = {}
            try:
                with torch.inference_mode():
                    for layer in metadata.layers:
                        dtype = getattr(torch, layer.dtype)
                        target = torch.empty(
                            (
                                len(metadata.original_ids),
                                layer.num_kv_heads,
                                2 * layer.head_size,
                            ),
                            dtype=dtype,
                            device="cpu",
                        )
                        payload[layer.name] = target
                        for lo in range(0, len(target), self.transfer_tokens):
                            count = min(self.transfer_tokens, len(target) - lo)
                            source = read_source(layer.name, lo, count)
                            if (
                                not isinstance(source, torch.Tensor)
                                or source.dtype != dtype
                                or source.shape != target[lo : lo + count].shape
                                or not torch.isfinite(source).all()
                            ):
                                raise ValueError(
                                    "Source KV shape, dtype, or values are invalid."
                                )
                            # Blocking D2H copy: publication never races a pending DMA.
                            target[lo : lo + count].copy_(
                                source.detach(), non_blocking=False
                            )
                self._entries[key] = (metadata, payload, charge)
                self._leases[key] = 0
            except BaseException:
                self._entries.pop(key, None)
                self._leases.pop(key, None)
                self.used_bytes -= charge
                payload.clear()
                raise

    def metadata(self, key: str) -> CheckpointMetadata:
        with self._lock:
            return self._entries[key][0]

    def create_encoded(self, metadata, reserved_bytes, build):
        """Reserve before encoding and atomically install one independent payload."""
        from dataclasses import replace

        _integer(reserved_bytes, "reserved_bytes", 1)
        with self._lock:
            key = metadata.checkpoint_id
            if key in self._entries:
                raise ValueError("Checkpoint identity already exists.")
            if self.used_bytes + reserved_bytes > self.max_bytes:
                raise MemoryError(
                    "CPU checkpoint encoding reservation exceeds memory budget."
                )
            self.used_bytes += reserved_bytes
            payload = None
            try:
                final, payload = build()
                if (
                    replace(final, transport_json=None) != metadata
                    or final.payload_bytes != payload.nbytes
                ):
                    raise ValueError(
                        "Encoded checkpoint metadata or byte count is inconsistent."
                    )
                charge = self.charged_bytes(final) + payload.ownership_bytes
                if charge > reserved_bytes:
                    raise MemoryError("Encoding exceeded its conservative reservation.")
                self._entries[key] = final, payload, charge
                self._leases[key] = 0
                self.used_bytes -= reserved_bytes - charge
            except BaseException:
                self._entries.pop(key, None)
                self._leases.pop(key, None)
                self.used_bytes -= reserved_bytes
                if payload is not None:
                    payload.clear()
                raise

    @contextmanager
    def lease(self, key: str) -> Iterator[_CheckpointReader]:
        with self._lock:
            metadata, payload, _ = self._entries[key]
            self._leases[key] += 1
            reader = (
                payload.reader(metadata, self.transfer_tokens)
                if hasattr(payload, "reader")
                else _CheckpointReader(metadata, payload, self.transfer_tokens)
            )
        try:
            yield reader
        finally:
            with self._lock:
                reader.active = False
                reader._payload = {}
                self._leases[key] -= 1

    def release(self, key: str) -> bool:
        with self._lock:
            if key not in self._entries:
                return False
            if self._leases[key]:
                raise RuntimeError("Cannot release an actively leased checkpoint.")
            _, payload, charge = self._entries.pop(key)
            payload.clear()
            del self._leases[key]
            self.used_bytes -= charge
            return True

    def clear(self) -> None:
        with self._lock:
            if any(self._leases.values()):
                raise RuntimeError("Cannot clear actively leased checkpoints.")
            for key in list(self._entries):
                self.release(key)


@dataclass(frozen=True)
class CheckpointCopy:
    source_start: int
    destination_start: int
    count: int


def checkpoint_copy_ranges(
    original_length: int, start: int, end: int, replacement_length: int, plan: Any
) -> tuple[CheckpointCopy, ...]:
    """Seed all prefix/reused rows, including pages Phase 1 would have shared."""
    for name, value in (
        ("original_length", original_length),
        ("start", start),
        ("end", end),
        ("replacement_length", replacement_length),
    ):
        _integer(value, name)
    if (
        not start <= end <= original_length
        or plan.suffix_start != start + replacement_length
        or plan.new_length != start + replacement_length + original_length - end
    ):
        raise ValueError("Checkpoint edit and plan do not match.")
    copies = [CheckpointCopy(0, 0, start)] if start else []
    for block in plan.reuse_blocks:
        lo = max(plan.suffix_start, block * plan.block_size)
        hi = min(plan.new_length, (block + 1) * plan.block_size)
        source = end + lo - plan.suffix_start
        if (
            copies
            and copies[-1].source_start + copies[-1].count == source
            and copies[-1].destination_start + copies[-1].count == lo
        ):
            previous = copies.pop()
            copies.append(
                CheckpointCopy(
                    previous.source_start,
                    previous.destination_start,
                    previous.count + hi - lo,
                )
            )
        else:
            copies.append(CheckpointCopy(source, lo, hi - lo))
    return tuple(copies)
