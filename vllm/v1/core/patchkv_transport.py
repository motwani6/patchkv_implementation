# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference mixed-format checkpoint creation and packed reusable-row transport."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

from vllm.v1.core.patchkv_precision import repair_policy_fingerprint
from vllm.v1.core.patchkv_quantization import CODEC, encode_kv_block, value_residual


class EncodedCheckpointPayload:
    def __init__(self, layers):
        self.layers = layers

    @property
    def nbytes(self):
        return sum(block.nbytes for blocks in self.layers.values() for block in blocks)

    @property
    def ownership_bytes(self):
        return 4096 * sum(len(blocks) for blocks in self.layers.values())

    def clear(self):
        self.layers.clear()

    def reader(self, metadata, transfer_tokens):
        return EncodedCheckpointReader(metadata, self, transfer_tokens)


class EncodedCheckpointReader:
    def __init__(self, metadata, payload, transfer_tokens):
        self.metadata, self._payload, self.transfer_tokens = (
            metadata,
            payload,
            transfer_tokens,
        )
        self.active = True
        self.transfer_bytes = 0
        self.format_bytes = {}

    def read(self, layer, start, count):
        return self.read_to(layer, start, count, "cpu")

    def read_to(self, layer, start, count, device):
        import torch

        if not self.active:
            raise ValueError("Checkpoint lease has ended.")
        if (
            type(start) is not int
            or type(count) is not int
            or start < 0
            or count < 0
            or count > self.transfer_tokens
            or start + count > len(self.metadata.original_ids)
        ):
            raise ValueError("Read exceeds checkpoint bounds or transfer chunk limit.")
        parts = []
        cursor = start
        while cursor < start + count:
            block_index, offset = divmod(cursor, self.metadata.block_size)
            block = self._payload.layers[layer][block_index]
            taken = min(start + count - cursor, block.count - offset)
            selected = block.slice(offset, taken)
            parts.append(selected.decode_to(device))
            self.transfer_bytes += selected.nbytes
            self.format_bytes[block.format] = (
                self.format_bytes.get(block.format, 0) + selected.nbytes
            )
            cursor += taken
        if not parts:
            spec = next(x for x in self.metadata.layers if x.name == layer)
            return torch.empty(
                0,
                spec.num_kv_heads,
                2 * spec.head_size,
                device=device,
                dtype=torch.float16,
            )
        return torch.cat(parts)


def validate_transport_prefix(metadata, prefix_tokens):
    if metadata.transport_json is not None:
        tags = json.loads(metadata.transport_json)["precision_tags"]
        if any(
            tags[b.logical_index] != "FP16"
            for b in metadata.blocks
            if b.source_start < prefix_tokens
        ):
            raise ValueError(
                "Unchanged prefix is quantized; protect more prefix during creation."
            )


def validate_transport_repair(metadata, profile, budget, mode):
    policy = json.loads(metadata.transport_json)["policy"]
    expected = policy.get("repair_fingerprint")
    if expected is not None and expected != repair_policy_fingerprint(
        asdict(profile) if profile is not None else None,
        asdict(budget) if budget is not None else None,
        mode,
    ):
        raise ValueError(
            "Precision policy was calibrated with a different semantic repair policy."
        )


def create_quantized_checkpoint(
    store, metadata, read_source, policy, protected_prefix_tokens
):
    """Freeze one tag per block using equations (13)/(14) or a fixed experiment."""
    import torch

    if metadata.transport_json is not None:
        raise ValueError(
            "Encode from uncompressed KV; repeated lossy encoding is unsupported."
        )
    if type(
        protected_prefix_tokens
    ) is not int or not 0 <= protected_prefix_tokens <= len(metadata.original_ids):
        raise ValueError("Invalid protected prefix length.")
    if any(layer.dtype != "float16" for layer in metadata.layers):
        raise ValueError(
            "Mixed FP16/K8 checkpoint storage currently requires an FP16 source cache."
        )
    policy.validate_context(metadata)
    if policy.mode != "fixed_experiment" and metadata.attention_json is None:
        raise ValueError(
            "Risk-based precision selection requires stored attention importance."
        )
    scores = (
        json.loads(metadata.attention_json)["scores"]
        if metadata.attention_json
        else None
    )
    policy_json = json.dumps(asdict(policy), allow_nan=False)
    worst = sum(
        b.count * layer.num_kv_heads * max(4 * layer.head_size, 2 * layer.head_size + 8)
        for b in metadata.blocks
        for layer in metadata.layers
    )
    reservation = (
        store.charged_bytes(metadata)
        - metadata.source_precision_bytes
        + worst
        + 8192 * (1 + len(metadata.blocks) * len(metadata.layers))
        + 2 * len(policy_json)
    )

    def build():
        payload = {layer.name: [] for layer in metadata.layers}
        tags, risks, errors4 = [], [], []
        for block in metadata.blocks:
            sources, candidates8, candidates4 = {}, {}, {}
            errors8_layers, errors4_layers = [], []
            protected = block.source_start < protected_prefix_tokens
            for layer in metadata.layers:
                chunks = []
                for offset in range(0, block.count, store.transfer_tokens):
                    count = min(store.transfer_tokens, block.count - offset)
                    chunk = read_source(layer.name, block.source_start + offset, count)
                    if (
                        chunk.dtype != torch.float16
                        or chunk.shape
                        != (count, layer.num_kv_heads, 2 * layer.head_size)
                        or not torch.isfinite(chunk).all()
                    ):
                        raise ValueError("Invalid FP16 source KV for encoding.")
                    chunks.append(chunk.detach().cpu())
                source = torch.cat(chunks)
                sources[layer.name] = source
                if not protected and policy.mode != "fixed_experiment":
                    eight = encode_kv_block(source, "K8/V8")
                    four = encode_kv_block(source, "K8/V4")
                    candidates8[layer.name], candidates4[layer.name] = eight, four
                    errors8_layers.append(
                        value_residual(source, eight.decode_to(dtype=torch.float32))
                    )
                    errors4_layers.append(
                        value_residual(source, four.decode_to(dtype=torch.float32))
                    )
            risk = error4 = None
            if protected:
                tag = "FP16"
            elif policy.mode == "fixed_experiment":
                tag = policy.fixed_format
            else:
                residual = torch.stack(errors8_layers).mean(0)
                importance = torch.tensor(
                    scores[block.source_start : block.source_start + block.count],
                    dtype=torch.float64,
                )
                z = torch.sqrt(importance * residual)
                risk = (policy.alpha * z.max() + (1 - policy.alpha) * z.mean()).item()
                error4 = torch.stack(errors4_layers).mean(0).max().item()
                tag = policy.choose(risk, error4)
            tags.append(tag)
            risks.append(risk)
            errors4.append(error4)
            for layer in metadata.layers:
                encoded = (
                    candidates8.get(layer.name)
                    if tag == "K8/V8"
                    else candidates4.get(layer.name)
                    if tag == "K8/V4"
                    else None
                )
                payload[layer.name].append(
                    encoded
                    if encoded is not None
                    else encode_kv_block(sources[layer.name], tag)
                )
        owned = EncodedCheckpointPayload(payload)
        transport = dict(
            codec=CODEC,
            precision_tags=tags,
            risks=risks,
            error4=errors4,
            protected_prefix_tokens=protected_prefix_tokens,
            policy=asdict(policy),
            quality_calibrated=policy.quality_calibrated,
            payload_bytes=owned.nbytes,
        )
        final = replace(
            metadata,
            transport_json=json.dumps(
                transport, allow_nan=False, separators=(",", ":")
            ),
        )
        return final, owned

    store.create_encoded(metadata, reservation, build)
