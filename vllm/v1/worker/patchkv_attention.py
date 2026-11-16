# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in attention capture during one uncached, in-process checkpoint pass."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from functools import partial
from typing import Any

import torch

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.v1.core.patchkv_attention import (
    AttentionAccumulator,
    AttentionLayerSpec,
    AttentionMetadata,
)


class CheckpointAttentionCollector:
    """Capture actual post-RoPE Q/K and reduce importance on CPU at checkpoint time.

    Use only for one uncached request with eager execution, TP/PP=1, and dense
    causal Qwen2/Llama attention. Query positions can include generated probes.
    No model parameters, outputs, KV pages, or production attention code change.
    Missing rows, prefix-cache hits, padded/batched execution, and unsupported
    attention features fail closed. Hooks are removed even after exceptions.
    """

    def __init__(
        self,
        runner: Any,
        original_ids: Sequence[int],
        *,
        model_key: str,
        checkpoint_id: str,
        block_size: int,
        layer_names: Sequence[str],
        query_positions: Sequence[int],
        key_tile: int = 256,
        query_tile: int = 2,
        head_tile: int = 8,
    ) -> None:
        config = runner.vllm_config
        parallel = config.parallel_config
        if (
            not original_ids
            or not query_positions
            or any(type(x) is not int or x < 0 for x in query_positions)
            or tuple(query_positions) != tuple(sorted(set(query_positions)))
        ):
            raise ValueError(
                "Need source tokens and sorted unique nonnegative query positions."
            )
        groups = runner.kv_cache_config.kv_cache_groups
        if len(groups) != 1 or groups[0].kv_cache_spec.block_size != block_size:
            raise ValueError(
                "Checkpoint block size must match the single execution cache group."
            )
        if (
            not config.model_config.enforce_eager
            or any(
                getattr(parallel, name) != 1
                for name in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                    "decode_context_parallel_size",
                    "prefill_context_parallel_size",
                )
            )
            or config.speculative_config is not None
            or config.lora_config is not None
            or config.kv_transfer_config is not None
        ):
            raise ValueError(
                "Capture requires eager, single-request, single-GPU execution."
            )
        self.model = runner.get_model()
        context = config.compilation_config.static_forward_context
        self.layers = {}
        specs = []
        for name in layer_names:
            layer = context[name]
            parent = self.model.get_submodule(name.rsplit(".", 1)[0])
            parent_type = f"{type(parent).__module__}.{type(parent).__name__}"
            rope = getattr(parent, "rotary_emb", None)
            impl = layer.impl
            if (
                parent_type
                not in (
                    "vllm.model_executor.models.qwen2.Qwen2Attention",
                    "vllm.model_executor.models.llama.LlamaAttention",
                )
                or self.model.get_submodule(name) is not layer
                or layer.attn_backend.get_name() != "TRITON_ATTN"
                or layer.attn_type != "decoder"
                or layer.sliding_window is not None
                or layer.kv_cache_dtype
                not in ("auto", "float16", "bfloat16", "float32")
                or getattr(layer, "kv_sharing_target_layer_name", None) is not None
                or getattr(layer, "has_sink", False)
                or getattr(layer, "use_mm_prefix", False)
                or getattr(impl, "alibi_slopes", None) is not None
                or getattr(impl, "sinks", None) is not None
                or getattr(impl, "logits_soft_cap", None) not in (None, 0)
                or getattr(impl, "sliding_window", (-1, -1)) != (-1, -1)
                or getattr(parent, "dual_chunk_attention_config", None)
                or not isinstance(rope, RotaryEmbedding)
                or type(rope).forward_native is not RotaryEmbedding.forward_native
                or rope.cos_sin_cache.ndim != 2
                or rope.cos_sin_cache.shape[1] != rope.rotary_dim
            ):
                raise ValueError(
                    f"Unsupported checkpoint attention configuration: {name}."
                )
            # Bind the standard RoPE layout and representative table rows. The
            # caller's model_key must also include revision and RoPE settings.
            last = max(query_positions) if query_positions else 0
            if last >= len(rope.cos_sin_cache):
                raise ValueError(
                    "Requested probe positions exceed the initialized RoPE table."
                )
            rope_info = {
                "class": f"{type(rope).__module__}.{type(rope).__name__}",
                "head_size": rope.head_size,
                "rotary_dim": rope.rotary_dim,
                "neox": rope.is_neox_style,
                "samples": rope.cos_sin_cache[[0, last]].float().cpu().tolist(),
            }
            signature = hashlib.sha256(
                json.dumps(rope_info, sort_keys=True).encode()
            ).hexdigest()
            specs.append(
                AttentionLayerSpec(
                    name,
                    layer.num_heads,
                    layer.num_kv_heads,
                    layer.head_size,
                    impl.scale,
                    signature,
                )
            )
            self.layers[name] = layer
        self.accumulator = AttentionAccumulator(
            original_ids,
            model_key=model_key,
            checkpoint_id=checkpoint_id,
            block_size=block_size,
            query_positions=query_positions,
            layers=specs,
            key_tile=key_tile,
            query_tile=query_tile,
            head_tile=head_tile,
        )
        self.original_ids = tuple(original_ids)
        self.next_position = 0
        self.positions = None
        self.pending: set[str] = set()
        self.handles: list[Any] = []
        self.metadata: AttentionMetadata | None = None
        self.started = False
        self.failed = False

    def _before_model(self, module, args, kwargs) -> None:
        if self.pending or self.metadata is not None:
            raise ValueError("Incomplete or already finalized attention capture.")
        positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
        ids = kwargs.get("input_ids", args[0] if args else None)
        if (
            not isinstance(positions, torch.Tensor)
            or positions.ndim != 1
            or positions.dtype not in (torch.int32, torch.int64)
            or not isinstance(ids, torch.Tensor)
            or ids.ndim != 1
            or len(ids) != len(positions)
            or not len(ids)
            or kwargs.get("inputs_embeds") is not None
        ):
            raise ValueError(
                "Capture requires one unpadded token request with 1D positions."
            )
        actual_positions = positions.detach().cpu().tolist()
        if actual_positions != list(
            range(self.next_position, self.next_position + len(ids))
        ):
            raise ValueError(
                "Cached, batched, or noncontiguous source pass cannot be captured."
            )
        known = max(0, min(len(ids), len(self.original_ids) - self.next_position))
        if (
            tuple(ids[:known].detach().cpu().tolist())
            != self.original_ids[self.next_position : self.next_position + known]
        ):
            raise ValueError("Checkpoint tokens do not match the captured request.")
        self.next_position += len(ids)
        self.positions = positions
        self.pending = set(self.layers)

    def _before_attention(self, name, module, args, kwargs) -> None:
        if name not in self.pending or self.positions is None:
            raise ValueError("Attention called outside its checkpoint forward.")
        query = kwargs.get("query", args[0] if args else None)
        keys = kwargs.get("key", args[1] if len(args) > 1 else None)
        if query is None or keys is None:
            raise ValueError("Checkpoint capture needs explicit post-RoPE Q/K.")
        spec = self.accumulator.specs[name]
        self.accumulator.record(
            name,
            self.positions,
            query.reshape(-1, spec.num_query_heads, spec.head_size),
            keys.reshape(-1, spec.num_kv_heads, spec.head_size),
        )
        self.pending.remove(name)

    def __enter__(self) -> CheckpointAttentionCollector:
        if self.started:
            raise ValueError("A checkpoint collector is single-use.")
        self.started = True
        try:
            self.handles.append(
                self.model.register_forward_pre_hook(
                    self._before_model, with_kwargs=True
                )
            )
            for name, layer in self.layers.items():
                self.handles.append(
                    layer.register_forward_pre_hook(
                        partial(self._before_attention, name), with_kwargs=True
                    )
                )
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.positions = None
        if exc_type is not None:
            self.failed = True
            self.accumulator.keys.clear()
            self.accumulator.queries.clear()

    def finalize(self) -> AttentionMetadata:
        if self.failed:
            raise ValueError("Cannot finalize an aborted checkpoint pass.")
        if self.metadata is not None:
            return self.metadata
        if self.pending or self.next_position < max(
            len(self.original_ids), self.accumulator.limit
        ):
            raise ValueError("Checkpoint/probe pass did not complete.")
        self.metadata = self.accumulator.finalize()
        self.accumulator.keys.clear()
        self.accumulator.queries.clear()
        return self.metadata
