# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-precision gather/restore for locally owned CPU KV checkpoints."""

from __future__ import annotations

import hashlib
import json

import torch

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.v1.core.patchkv import plan_suffix_repair
from vllm.v1.core.patchkv_checkpoint import CheckpointLayer, checkpoint_copy_ranges
from vllm.v1.worker.kv_cache_edit import _prefill, _relocate_keys


def checkpoint_layers(runner, original_length: int, block_size: int):
    """Validate the supported resident layout and bind native RoPE parameters."""
    model = runner.get_model()
    context = runner.vllm_config.compilation_config.static_forward_context
    result = {}
    specs = []
    for group in runner.attn_groups[0]:
        if getattr(group, "kernel_block_stride", 0):
            raise ValueError("Checkpoints do not support packed kernel-block strides.")
        for name in group.layer_names:
            layer = context[name]
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            rope = getattr(parent, "rotary_emb", None)
            cache = layer.kv_cache
            if (
                not isinstance(rope, RotaryEmbedding)
                or type(rope).forward_native is not RotaryEmbedding.forward_native
                or rope.cos_sin_cache.ndim != 2
                or rope.cos_sin_cache.shape[1] != rope.rotary_dim
                or cache.ndim != 4
                or cache.shape[1] != layer.num_kv_heads
                or cache.shape[-1] != 2 * rope.head_size
                or cache.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or block_size % cache.shape[2]
                or layer.attn_backend.get_name() != "TRITON_ATTN"
                or getattr(layer, "kv_sharing_target_layer_name", None) is not None
            ):
                raise ValueError(f"Unsupported checkpoint KV/RoPE layout: {name}.")
            last = max(0, original_length - 1)
            if last >= len(rope.cos_sin_cache):
                raise ValueError("Checkpoint source positions exceed the RoPE table.")
            signature = json.dumps(
                {
                    "class": f"{type(rope).__module__}.{type(rope).__name__}",
                    "head_size": rope.head_size,
                    "rotary_dim": rope.rotary_dim,
                    "neox": rope.is_neox_style,
                    "table": rope.cos_sin_cache[[0, last]].float().cpu().tolist(),
                },
                sort_keys=True,
            )
            specs.append(
                CheckpointLayer(
                    name,
                    layer.num_kv_heads,
                    rope.head_size,
                    str(cache.dtype).removeprefix("torch."),
                    hashlib.sha256(signature.encode()).hexdigest(),
                )
            )
            result[name] = cache.transpose(1, 2), rope
    if not specs:
        raise ValueError("Checkpoint requires at least one attention layer.")
    return result, tuple(specs)


def gather_checkpoint_rows(
    cache, source_pages, block_size: int, start: int, count: int
):
    """Gather one bounded logical interval; physical IDs never enter storage."""
    positions = torch.arange(start, start + count, device=cache.device)
    pages = torch.as_tensor(source_pages, dtype=torch.long, device=cache.device)
    slots = pages[positions // block_size] * block_size + positions % block_size
    kernel_size = cache.shape[1]
    return cache[slots // kernel_size, slots % kernel_size]


@torch.inference_mode()
def restore_checkpoint_kv(
    runner, checkpoint, destination, start, end, replacement_ids, plan
) -> None:
    """Seed every reused dependency, then prefill mandatory and repair intervals."""
    metadata = checkpoint.metadata
    block_size = metadata.block_size
    expected = plan_suffix_repair(
        len(metadata.original_ids),
        start,
        end,
        len(replacement_ids),
        block_size,
        plan.repair_blocks,
    )
    if expected != plan:
        raise ValueError("Checkpoint recovery plan is inconsistent.")
    layers, specs = checkpoint_layers(runner, len(metadata.original_ids), block_size)
    if specs != metadata.layers:
        raise ValueError(
            "Checkpoint KV geometry, dtype, or RoPE configuration changed."
        )
    if (
        len(destination) != (plan.new_length + block_size - 1) // block_size
        or len(set(destination)) != len(destination)
        or any(type(x) is not int or x < 0 for x in destination)
    ):
        raise ValueError("Restore needs a private page for each destination block.")
    pages = torch.tensor(destination, dtype=torch.long, device=runner.device)
    copies = checkpoint_copy_ranges(
        len(metadata.original_ids), start, end, len(replacement_ids), plan
    )
    for name, (cache, rope) in layers.items():
        if (
            destination
            and (max(destination) + 1) * block_size > cache.shape[0] * cache.shape[1]
        ):
            raise ValueError("Destination page exceeds physical cache capacity.")
        if plan.new_length > len(rope.cos_sin_cache):
            raise ValueError("Restored positions exceed the initialized RoPE table.")
        for mapping in copies:
            for offset in range(0, mapping.count, checkpoint.transfer_tokens):
                count = min(checkpoint.transfer_tokens, mapping.count - offset)
                source_pos = mapping.source_start + offset
                destination_pos = mapping.destination_start + offset
                rows = checkpoint.read_to(name, source_pos, count, cache.device)
                old = torch.arange(source_pos, source_pos + count, device=cache.device)
                new = torch.arange(
                    destination_pos, destination_pos + count, device=cache.device
                )
                if source_pos != destination_pos:
                    _relocate_keys(rows[..., : rope.head_size], rope, old, new)
                slots = pages[new // block_size] * block_size + new % block_size
                kernel_size = cache.shape[1]
                cache[slots // kernel_size, slots % kernel_size] = rows
    # All layers' prefix and reused suffix KV must be placed before any forward.
    _prefill(runner, replacement_ids, start, pages, block_size)
    suffix = metadata.original_ids[end:]
    for lo, hi in plan.repair_ranges:
        _prefill(
            runner,
            list(suffix[lo - plan.suffix_start : hi - plan.suffix_start]),
            lo,
            pages,
            block_size,
        )
