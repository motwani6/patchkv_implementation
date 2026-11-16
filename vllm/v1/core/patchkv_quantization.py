# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference FP16/K8V8/K8V4 codec; active attention caches stay floating point."""

from __future__ import annotations

import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMATS = ("FP16", "K8/V8", "K8/V4")
CODEC = "symmetric_token_head_le_v1"


def pack_int4(values):
    """Pack signed nibbles along the last axis, low nibble first, zero padding."""
    import torch

    if values.dtype != torch.int8 or values.ndim == 0 or not values.shape[-1]:
        raise ValueError("INT4 input must be an int8 tensor with a nonempty last axis.")
    if ((values < -8) | (values > 7)).any():
        raise ValueError("Signed INT4 values must be in [-8, 7].")
    code = values.to(torch.uint8) & 15
    if code.shape[-1] % 2:
        code = torch.cat((code, torch.zeros_like(code[..., :1])), -1)
    return code[..., ::2] | (code[..., 1::2] << 4)


def unpack_int4(packed, dimension: int):
    import torch

    if (
        packed.dtype != torch.uint8
        or type(dimension) is not int
        or dimension <= 0
        or packed.ndim == 0
        or packed.shape[-1] != (dimension + 1) // 2
    ):
        raise ValueError("Packed INT4 shape/dimension mismatch.")
    if dimension % 2 and ((packed[..., -1] >> 4) != 0).any():
        raise ValueError("INT4 padding nibble must be zero.")
    codes = (
        torch.stack((packed & 15, packed >> 4), -1)
        .flatten(-2)[..., :dimension]
        .to(torch.int8)
    )
    return torch.where(codes >= 8, codes - 16, codes)


def _quantize(values, maximum):
    import torch

    values = values.float()
    extent = values.abs().amax(-1, keepdim=True)
    scale = torch.where(
        extent > 0,
        (extent / maximum).clamp_min(torch.finfo(torch.float32).tiny),
        torch.ones_like(extent),
    )
    codes = torch.round(values / scale).clamp(-maximum, maximum).to(torch.int8)
    return codes.contiguous(), scale.contiguous()


@dataclass(frozen=True)
class EncodedKVBlock:
    format: str
    count: int
    heads: int
    dimension: int
    tensors: tuple[Any, ...]

    def validate(self):
        import torch

        if self.format not in FORMATS or any(
            type(x) is not int or x <= 0
            for x in (self.count, self.heads, self.dimension)
        ):
            raise ValueError("Invalid encoded KV block header.")
        shape = self.count, self.heads, self.dimension
        expected = (
            ((shape[:-1] + (2 * self.dimension,), torch.float16),)
            if self.format == "FP16"
            else (
                (shape, torch.int8),
                (
                    shape[:-1]
                    + (
                        (
                            (self.dimension + 1) // 2
                            if self.format == "K8/V4"
                            else self.dimension
                        ),
                    ),
                    torch.uint8 if self.format == "K8/V4" else torch.int8,
                ),
                (shape[:-1] + (1,), torch.float32),
                (shape[:-1] + (1,), torch.float32),
            )
        )
        if len(expected) != len(self.tensors):
            raise ValueError("Encoded KV tensor count mismatch.")
        for tensor, (wanted_shape, dtype) in zip(self.tensors, expected):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tuple(tensor.shape) != wanted_shape
                or tensor.dtype != dtype
                or tensor.requires_grad
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "Encoded KV tensors must have the declared CPU layout."
                )
            if dtype.is_floating_point and not torch.isfinite(tensor).all():
                raise ValueError("Encoded FP16/scales must be finite.")
        if self.format != "FP16" and any(
            not (scale > 0).all() for scale in self.tensors[2:]
        ):
            raise ValueError("Quantization scales must be positive.")
        if (
            self.format == "K8/V4"
            and self.dimension % 2
            and (self.tensors[1][..., -1] >> 4).any()
        ):
            raise ValueError("Invalid INT4 padding.")

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)

    def slice(self, start, count):
        if (
            type(start) is not int
            or type(count) is not int
            or start < 0
            or count <= 0
            or start + count > self.count
        ):
            raise ValueError("Invalid encoded block slice.")
        return EncodedKVBlock(
            self.format,
            count,
            self.heads,
            self.dimension,
            tuple(t[start : start + count].clone() for t in self.tensors),
        )

    def decode_to(self, device="cpu", dtype=None):
        """Transfer packed codes/scales first; dequantize on the destination."""
        import torch

        dtype = torch.float16 if dtype is None else dtype
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Destination cache dtype must be floating point.")
        data = tuple(t.to(device=device, non_blocking=False) for t in self.tensors)
        if self.format == "FP16":
            return data[0].to(dtype=dtype).clone()
        keys, values, k_scale, v_scale = data
        if self.format == "K8/V4":
            values = unpack_int4(values, self.dimension)
        return torch.cat((keys.float() * k_scale, values.float() * v_scale), -1).to(
            dtype=dtype
        )

    def to_dict(self):
        self.validate()
        if sys.byteorder != "little":
            raise ValueError("Reference serialization requires a little-endian host.")
        import torch

        return dict(
            codec=CODEC,
            format=self.format,
            count=self.count,
            heads=self.heads,
            dimension=self.dimension,
            data=[
                base64.b64encode(bytes(t.view(torch.uint8).flatten().tolist())).decode(
                    "ascii"
                )
                for t in self.tensors
            ],
        )

    @classmethod
    def from_dict(cls, data, max_bytes=64 << 20):
        import torch

        if data.get("codec") != CODEC or sys.byteorder != "little":
            raise ValueError("Unsupported codec or host byte order.")
        fmt, n, h, d = (data[k] for k in ("format", "count", "heads", "dimension"))
        if fmt not in FORMATS or any(type(x) is not int or x <= 0 for x in (n, h, d)):
            raise ValueError("Invalid serialized KV block shape.")
        shapes = (
            [(n, h, 2 * d)]
            if fmt == "FP16"
            else [
                (n, h, d),
                (n, h, (d + 1) // 2 if fmt == "K8/V4" else d),
                (n, h, 1),
                (n, h, 1),
            ]
        )
        dtypes = (
            [torch.float16]
            if fmt == "FP16"
            else [
                torch.int8,
                torch.uint8 if fmt == "K8/V4" else torch.int8,
                torch.float32,
                torch.float32,
            ]
        )
        import math

        sizes = [
            math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            for shape, dtype in zip(shapes, dtypes)
        ]
        if sum(sizes) > max_bytes or len(data["data"]) != len(sizes):
            raise ValueError(
                "Serialized block exceeds byte limit or has invalid tensor count."
            )
        tensors = []
        for encoded, size, shape, dtype in zip(data["data"], sizes, shapes, dtypes):
            if len(encoded) != 4 * ((size + 2) // 3):
                raise ValueError("Encoded byte length does not match block shape.")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) != size:
                raise ValueError("Invalid encoded payload length.")
            tensors.append(
                torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape)
            )
        result = cls(fmt, n, h, d, tuple(tensors))
        result.validate()
        return result

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), allow_nan=False), encoding="utf-8"
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path, max_bytes=64 << 20):
        if Path(path).stat().st_size > 2 * max_bytes + 4096:
            raise ValueError("Encoded block file exceeds the byte limit.")
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8")), max_bytes
        )


def encode_kv_block(rows, format: str) -> EncodedKVBlock:
    import torch

    if (
        format not in FORMATS
        or not isinstance(rows, torch.Tensor)
        or rows.ndim != 3
        or not all(rows.shape)
        or rows.shape[-1] % 2
        or not rows.is_floating_point()
        or not torch.isfinite(rows).all()
    ):
        raise ValueError("Expected finite floating-point [tokens, heads, 2D] KV.")
    rows = rows.detach().cpu()
    if format == "FP16":
        result = EncodedKVBlock(
            format,
            len(rows),
            rows.shape[1],
            rows.shape[2] // 2,
            (rows.half().contiguous().clone(),),
        )
    else:
        keys, values = rows.chunk(2, -1)
        k, ks = _quantize(keys, 127)
        v, vs = _quantize(values, 7 if format == "K8/V4" else 127)
        result = EncodedKVBlock(
            format,
            len(rows),
            rows.shape[1],
            rows.shape[2] // 2,
            (k, pack_int4(v) if format == "K8/V4" else v, ks, vs),
        )
    result.validate()
    return result


def value_residual(rows, reconstructed):
    """Per-token normalized squared value error; not a generation-quality test."""
    import torch

    original = rows[..., rows.shape[-1] // 2 :].double().flatten(1)
    restored = reconstructed[..., reconstructed.shape[-1] // 2 :].double().flatten(1)
    numerator = (original - restored).square().sum(1)
    denominator = original.square().sum(1)
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1e-300),
        torch.where(numerator == 0, 0, torch.inf),
    )
