# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Idle in-process LLM checkpoint ownership and restoration transactions."""

from __future__ import annotations

import json
from dataclasses import asdict
from threading import RLock
from uuid import uuid4

import torch

from vllm.config import set_current_vllm_config
from vllm.v1.core.patchkv import (
    plan_attention_guided_suffix_repair,
    plan_profiled_suffix_repair,
    plan_suffix_repair,
)
from vllm.v1.core.patchkv_attention import AttentionMetadata
from vllm.v1.core.patchkv_checkpoint import (
    CheckpointBlock,
    CheckpointMetadata,
    CPUCheckpointStore,
)
from vllm.v1.core.patchkv_transport import (
    create_quantized_checkpoint,
    validate_transport_prefix,
    validate_transport_repair,
)
from vllm.v1.worker.patchkv_checkpoint import (
    checkpoint_layers,
    gather_checkpoint_rows,
    restore_checkpoint_kv,
)


class KVCheckpointHandle:
    """Explicitly released CPU ownership; carries no GPU page references."""

    def __init__(self, session, checkpoint_id):
        self._session = session
        self.checkpoint_id = checkpoint_id

    @property
    def metadata(self):
        return self._session.store.metadata(self.checkpoint_id)

    def release(self) -> bool:
        released = self._session.store.release(self.checkpoint_id)
        if released:
            self._session._restore_statistics.pop(self.checkpoint_id, None)
        return released

    @property
    def restore_statistics(self):
        self._session.store.metadata(self.checkpoint_id)
        return json.loads(
            json.dumps(self._session._restore_statistics.get(self.checkpoint_id, {}))
        )

    def compress(self, precision_policy, *, protected_prefix_tokens=0):
        """Create an independent encoded snapshot from an uncompressed CPU source."""
        return self._session.compress(
            self.checkpoint_id, precision_policy, protected_prefix_tokens
        )

    def restore(self, llm, original_ids, replacement_ids, start, end, **options):
        if llm is not self._session.llm:
            raise ValueError("Checkpoint handle belongs to a different LLM instance.")
        return self._session.restore(
            self.checkpoint_id, original_ids, replacement_ids, start, end, **options
        )

    def __enter__(self):
        self._session.store.metadata(self.checkpoint_id)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


class KVCheckpointSession:
    """Single-engine memory budget; callers must serialize generate/edit calls."""

    def __init__(self, llm, max_cpu_bytes: int):
        client = llm.llm_engine.engine_core
        core = getattr(client, "engine_core", None)
        if core is None:
            raise ValueError(
                "CPU checkpoints require VLLM_ENABLE_V1_MULTIPROCESSING=0."
            )
        editor = getattr(core.scheduler.kv_cache_manager, "editor", None)
        parallel = core.vllm_config.parallel_config
        if editor is None or any(
            getattr(parallel, name) != 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        ):
            raise ValueError("Checkpoints require a single-GPU dense causal KV editor.")
        self.llm, self.core, self.editor = llm, core, editor
        self.worker = getattr(core.model_executor, "driver_worker", None)
        self.runner = getattr(self.worker, "model_runner", None)
        if self.runner is None or not callable(
            getattr(self.worker, "synchronize_device", None)
        ):
            raise ValueError("Checkpoints require a local UniProcExecutor GPU worker.")
        self.store = CPUCheckpointStore(max_cpu_bytes)
        self._lock = RLock()
        self._restore_statistics = {}

    def _idle(self):
        if self.core.scheduler.has_unfinished_requests():
            raise RuntimeError("Checkpoint operations require an idle engine.")

    def close(self):
        with self._lock:
            self._idle()
            self.store.clear()
            self._restore_statistics.clear()

    def _configuration(self, specs):
        model = self.core.vllm_config.model_config
        return json.dumps(
            {
                "model": model.model,
                "revision": model.revision,
                "commit": getattr(model.hf_config, "_commit_hash", None),
                "dtype": str(model.dtype),
                "max_model_len": model.max_model_len,
                "hf_config": model.hf_config.to_dict(),
                "layers": [asdict(s) for s in specs],
                "backend": "TRITON_ATTN",
                "block_size": self.editor.block_size,
            },
            sort_keys=True,
            allow_nan=False,
        )

    def create(
        self,
        original_ids,
        *,
        cache_salt,
        model_key,
        attention=None,
        precision_policy=None,
        protected_prefix_tokens=0,
    ):
        with self._lock:
            self._idle()
            original_ids = tuple(original_ids)
            config = self.core.vllm_config.model_config
            if len(original_ids) > config.max_model_len or any(
                type(t) is not int or not 0 <= t < config.get_vocab_size()
                for t in original_ids
            ):
                raise ValueError("Checkpoint source tokens exceed model limits.")
            layers, specs = checkpoint_layers(
                self.runner, len(original_ids), self.editor.block_size
            )
            attention_json = None
            if attention is not None:
                attention.validate_context(
                    original_ids,
                    model_key=model_key,
                    block_size=self.editor.block_size,
                    checkpoint_id=cache_salt,
                )
                attention_json = json.dumps(asdict(attention), allow_nan=False)
            key = "patchkv-" + uuid4().hex
            size = self.editor.block_size
            metadata = CheckpointMetadata(
                key,
                model_key,
                self._configuration(specs),
                original_ids,
                cache_salt,
                size,
                specs,
                tuple(
                    CheckpointBlock(i // size, i, min(size, len(original_ids) - i))
                    for i in range(0, len(original_ids), size)
                ),
                attention_json,
            )
            request = self.editor._request(list(original_ids), cache_salt)
            source = []
            for lo in range(0, len(original_ids), size):
                hit = self.editor.pool.get_cached_block(
                    self.editor._hash_at(request, min(lo + size, len(original_ids))),
                    [0],
                )
                if hit is None:
                    raise ValueError(
                        "Original KV must be resident during checkpoint creation."
                    )
                source.append(hit[0])
            self.editor.pool.touch(source)
            created = False
            try:
                self.worker.synchronize_device()
                source_pages = None

                def read(name, lo, count):
                    nonlocal source_pages
                    if source_pages is None:
                        source_pages = torch.tensor(
                            [b.block_id for b in source],
                            dtype=torch.long,
                            device=self.runner.device,
                        )
                    return gather_checkpoint_rows(
                        layers[name][0], source_pages, size, lo, count
                    )

                try:
                    if precision_policy is None:
                        if protected_prefix_tokens:
                            raise ValueError(
                                "Prefix protection is an encoded-checkpoint option."
                            )
                        self.store.create(metadata, read)
                    else:
                        create_quantized_checkpoint(
                            self.store,
                            metadata,
                            read,
                            precision_policy,
                            protected_prefix_tokens,
                        )
                    created = True
                finally:
                    self.worker.synchronize_device()
                return KVCheckpointHandle(self, key)
            except BaseException:
                if created:
                    self.store.release(key)
                raise
            finally:
                self.editor.pool.free_blocks(reversed(source))

    def compress(self, key, policy, protected_prefix_tokens):
        with self._lock, self.store.lease(key) as source:
            self._idle()
            from dataclasses import replace

            metadata = replace(source.metadata, checkpoint_id="patchkv-" + uuid4().hex)
            create_quantized_checkpoint(
                self.store, metadata, source.read, policy, protected_prefix_tokens
            )
            return KVCheckpointHandle(self, metadata.checkpoint_id)

    def restore(
        self,
        key,
        original_ids,
        replacement_ids,
        start,
        end,
        *,
        cache_salt=None,
        recompute_suffix=False,
        repair_suffix_blocks=None,
        drift_profile=None,
        attention_budget=None,
    ):
        with self._lock, self.store.lease(key) as checkpoint:
            self._idle()
            metadata = checkpoint.metadata
            size = self.editor.block_size
            _, specs = checkpoint_layers(self.runner, len(metadata.original_ids), size)
            metadata.validate(
                model_key=metadata.model_key,
                model_config_json=self._configuration(specs),
                layers=specs,
                original_ids=tuple(original_ids),
                block_size=size,
            )
            if cache_salt is not None and cache_salt != metadata.source_cache_salt:
                raise ValueError(
                    "Source cache salt does not match checkpoint provenance."
                )
            config = self.core.vllm_config.model_config
            if any(
                type(t) is not int or not 0 <= t < config.get_vocab_size()
                for t in replacement_ids
            ):
                raise ValueError("Replacement tokens are outside model vocabulary.")
            if (
                recompute_suffix
                and repair_suffix_blocks is not None
                or drift_profile is not None
                and (recompute_suffix or repair_suffix_blocks is not None)
                or attention_budget is not None
                and drift_profile is None
            ):
                raise ValueError(
                    "Choose direct, full, manual, or predicted checkpoint repair."
                )
            base = plan_suffix_repair(
                len(original_ids), start, end, len(replacement_ids), size, ()
            )
            if base.new_length > config.max_model_len:
                raise ValueError("Edited checkpoint exceeds max_model_len.")
            if metadata.transport_json is not None:
                validate_transport_prefix(metadata, start)
                if (
                    not recompute_suffix
                    and json.loads(metadata.transport_json)["policy"].get(
                        "repair_fingerprint"
                    )
                    is not None
                ):
                    validate_transport_repair(
                        metadata,
                        drift_profile,
                        attention_budget,
                        "manual" if repair_suffix_blocks is not None else "direct",
                    )
            if attention_budget is not None:
                if metadata.attention_json is None:
                    raise ValueError("Checkpoint has no attention importance metadata.")
                attention = AttentionMetadata.from_dict(
                    json.loads(metadata.attention_json)
                )
                plan = plan_attention_guided_suffix_repair(
                    original_ids,
                    start,
                    end,
                    len(replacement_ids),
                    size,
                    drift_profile,
                    attention,
                    attention_budget,
                    model_key=metadata.model_key,
                    checkpoint_id=metadata.source_cache_salt,
                )
            elif drift_profile is not None:
                plan = plan_profiled_suffix_repair(
                    len(original_ids),
                    start,
                    end,
                    len(replacement_ids),
                    size,
                    drift_profile,
                    model_key=metadata.model_key,
                )
            else:
                selected = (
                    range(
                        base.suffix_start // size, (base.new_length + size - 1) // size
                    )
                    if recompute_suffix and base.suffix_start < base.new_length
                    else repair_suffix_blocks or ()
                )
                plan = plan_suffix_repair(
                    len(original_ids), start, end, len(replacement_ids), size, selected
                )
            edited_ids = (
                list(original_ids[:start])
                + list(replacement_ids)
                + list(original_ids[end:])
            )
            salt = self.editor.salt_prefix + uuid4().hex
            request = self.editor._request(edited_ids, salt)
            allocated = []
            published = False
            try:
                allocated = self.editor.pool.get_new_blocks(
                    (plan.new_length + size - 1) // size
                )
                self.worker.synchronize_device()
                try:
                    with set_current_vllm_config(self.core.vllm_config):
                        restore_checkpoint_kv(
                            self.runner,
                            checkpoint,
                            [b.block_id for b in allocated],
                            start,
                            end,
                            list(replacement_ids),
                            plan,
                        )
                finally:
                    # No destination pages can be published/recycled before completion.
                    self.worker.synchronize_device()
                for i, block in enumerate(allocated):
                    self.editor._publish(
                        request, min((i + 1) * size, plan.new_length), block
                    )
                resend = plan.new_length - 1
                if resend > 0 and resend % size:
                    self.editor._publish(request, resend, allocated[resend // size])
                published = True
            finally:
                if not published:
                    for block in allocated:
                        self.editor.pool._maybe_evict_cached_block(block)
                self.editor.pool.free_blocks(reversed(allocated))
            self._restore_statistics[key] = dict(
                transfer_bytes=checkpoint.transfer_bytes,
                format_bytes=dict(checkpoint.format_bytes),
                checkpoint_payload_bytes=metadata.payload_bytes,
                source_precision_bytes=metadata.source_precision_bytes,
                repaired_blocks=list(plan.repair_blocks),
            )
            return salt
