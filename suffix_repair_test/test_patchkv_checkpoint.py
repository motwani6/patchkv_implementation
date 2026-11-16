# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-owned checkpoint lifecycle and real restore logic on synthetic paged KV."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from suffix_repair_test import test_patchkv_profile as profile_tests
from suffix_repair_test.patchkv_profile_tools import (
    load_attention_module,
    load_checkpoint_module,
)
from suffix_repair_test.test_patchkv_execution import load_definitions, load_method
from suffix_repair_test.test_patchkv_planner import planner

c = load_checkpoint_module()
a = load_attention_module()


def metadata(count=13, key="checkpoint", dtype="float32"):
    return c.CheckpointMetadata(
        key,
        "synthetic/model@revision:fp16",
        '{"model":"synthetic"}',
        tuple(range(count)),
        "source-salt",
        4,
        (c.CheckpointLayer("layer", 2, 4, dtype, "rope-id"),),
        tuple(
            c.CheckpointBlock(i // 4, i, min(4, count - i)) for i in range(0, count, 4)
        ),
    )


class CheckpointMetadataTest(unittest.TestCase):
    def test_metadata_roundtrip_logical_positions_and_compatibility(self):
        data = metadata()
        self.assertEqual(data.blocks[-1], c.CheckpointBlock(3, 12, 1))
        self.assertEqual(data.payload_bytes, 13 * 2 * 8 * 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            data.save(path)
            self.assertEqual(data, c.CheckpointMetadata.load(path))
            self.assertNotIn("physical", path.read_text())
        kwargs = dict(
            model_key=data.model_key,
            model_config_json=data.model_config_json,
            layers=data.layers,
            original_ids=data.original_ids,
            block_size=4,
        )
        data.validate(**kwargs)
        for changes in (
            dict(model_key="wrong"),
            dict(model_config_json='{"model":"other"}'),
            dict(original_ids=tuple(range(12))),
            dict(block_size=8),
            dict(layers=(replace(data.layers[0], rope_signature="different"),)),
        ):
            with self.assertRaises(ValueError):
                data.validate(**{**kwargs, **changes})

    def test_bad_mapping_schema_and_quantized_dtype_rejected(self):
        for changes in (
            dict(schema_version=2),
            dict(block_size=0),
            dict(blocks=()),
            dict(original_ids=(True,)),
            dict(model_config_json="[]"),
        ):
            with self.assertRaises(ValueError):
                replace(metadata(), **changes)
        with self.assertRaises(ValueError):
            c.CheckpointLayer("layer", 2, 4, "int8", "rope")

    def test_exhaustive_checkpoint_seed_mapping(self):
        for length in range(7):
            for start in range(length + 1):
                for end in range(start, length + 1):
                    for replacement in range(4):
                        suffix_start = start + replacement
                        new_length = suffix_start + length - end
                        blocks = sorted(
                            {i // 3 for i in range(suffix_start, new_length)}
                        )
                        for selected in ([], blocks, blocks[::2]):
                            plan = planner.plan_suffix_repair(
                                length, start, end, replacement, 3, selected
                            )
                            copies = c.checkpoint_copy_ranges(
                                length, start, end, replacement, plan
                            )
                            actual = {}
                            for mapping in copies:
                                for offset in range(mapping.count):
                                    old, new = (
                                        mapping.source_start + offset,
                                        mapping.destination_start + offset,
                                    )
                                    self.assertNotIn(new, actual)
                                    self.assertTrue(0 <= old < length)
                                    actual[new] = old
                            writes = set(range(start, suffix_start))
                            writes.update(
                                i
                                for lo, hi in plan.repair_ranges
                                for i in range(lo, hi)
                            )
                            self.assertFalse(set(actual) & writes)
                            self.assertEqual(
                                set(actual) | writes, set(range(new_length))
                            )
                            for new, old in actual.items():
                                self.assertEqual(
                                    old,
                                    new if new < start else end + new - suffix_start,
                                )


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch unavailable")
class CheckpointStoreTest(unittest.TestCase):
    def test_owned_storage_precision_leases_release_and_read_bounds(self):
        import torch

        for dtype in ("float16", "bfloat16", "float32"):
            data = metadata(dtype=dtype)
            original = torch.arange(13 * 16).reshape(13, 2, 8).to(getattr(torch, dtype))
            store = c.CPUCheckpointStore(1 << 20, transfer_tokens=3)
            store.create(
                data, lambda name, lo, count, source=original: source[lo : lo + count]
            )
            before = original.clone()
            original.zero_()
            with store.lease(data.checkpoint_id) as reader:
                torch.testing.assert_close(
                    reader.read("layer", 0, 3), before[:3], atol=0, rtol=0
                )
                changed = reader.read("layer", 0, 3)
                changed.zero_()
                torch.testing.assert_close(
                    reader.read("layer", 0, 3), before[:3], atol=0, rtol=0
                )
                with self.assertRaises(RuntimeError):
                    store.release(data.checkpoint_id)
                with self.assertRaises(ValueError):
                    reader.read("layer", 0, 4)
            with self.assertRaises(ValueError):
                reader.read("layer", 0, 1)
            self.assertTrue(store.release(data.checkpoint_id))
            self.assertFalse(store.release(data.checkpoint_id))
            self.assertEqual(store.used_bytes, 0)
            with self.assertRaises(KeyError):
                store.metadata(data.checkpoint_id)

    def test_budget_duplicate_and_failed_creation_rollback(self):
        import torch

        data = metadata()
        charge = c.CPUCheckpointStore.charged_bytes(data)
        reader = Mock(return_value=torch.ones(13, 2, 8))
        store = c.CPUCheckpointStore(charge - 1)
        with self.assertRaises(MemoryError):
            store.create(data, reader)
        reader.assert_not_called()
        store = c.CPUCheckpointStore(charge, transfer_tokens=4)
        calls = []

        def failing(name, lo, count):
            calls.append(lo)
            if lo:
                raise RuntimeError("copy failed")
            return torch.ones(count, 2, 8)

        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            store.create(data, failing)
        self.assertEqual(store.used_bytes, 0)
        store.create(data, lambda name, lo, count: torch.ones(count, 2, 8))
        self.assertEqual(store.used_bytes, charge)
        with self.assertRaises(ValueError):
            store.create(data, reader)
        with self.assertRaises(MemoryError):
            store.create(replace(data, checkpoint_id="second"), reader)
        store.release(data.checkpoint_id)
        self.assertEqual(store.used_bytes, 0)

    def test_empty_checkpoint_and_invalid_source_tensor(self):
        import torch

        store = c.CPUCheckpointStore(1 << 20)
        empty = metadata(count=0)
        reader = Mock()
        store.create(empty, reader)
        reader.assert_not_called()
        store.release(empty.checkpoint_id)
        for tensor in (
            torch.ones(13, 2, 8).half(),
            torch.ones(13, 1, 8),
            torch.full((13, 2, 8), float("nan")),
        ):
            with self.assertRaises(ValueError):
                store.create(metadata(), lambda *args, tensor=tensor: tensor)
            self.assertEqual(store.used_bytes, 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch unavailable")
class CheckpointRecoveryTest(unittest.TestCase):
    def test_synchronization_failure_rolls_back_completed_cpu_copy(self):
        env = self.environment()
        env.session.worker.synchronize_device = Mock(
            side_effect=[None, RuntimeError("sync failed")]
        )
        with self.assertRaisesRegex(RuntimeError, "sync failed"):
            env.session.create(
                list(range(13)), cache_salt="source-salt", model_key="model"
            )
        self.assertEqual(env.session.store.used_bytes, 0)
        self.assertTrue(all(page.ref_cnt == 0 for page in env.pool.source))

    def test_creation_copy_failure_and_budget_exhaustion_release_source_pins(self):

        env = self.environment()
        env.session.store = c.CPUCheckpointStore(1)
        with self.assertRaises(MemoryError):
            env.session.create(
                list(range(13)), cache_salt="source-salt", model_key="model"
            )
        self.assertTrue(all(page.ref_cnt == 0 for page in env.pool.source))
        self.assertEqual(env.session.store.used_bytes, 0)
        env.session.store = c.CPUCheckpointStore(1 << 20)
        env.context["layer1.attn"].kv_cache.fill_(float("nan"))
        with self.assertRaises(ValueError):
            env.session.create(
                list(range(13)), cache_salt="source-salt", model_key="model"
            )
        self.assertEqual(env.events[-1], ("sync",))
        self.assertTrue(all(page.ref_cnt == 0 for page in env.pool.source))
        self.assertEqual(env.session.store.used_bytes, 0)

    def test_model_configuration_and_rope_mismatch_rejected_before_allocation(self):
        for change in ("model", "rope"):
            env = self.environment()
            handle = env.session.create(
                list(range(13)), cache_salt="source-salt", model_key="model"
            )
            if change == "model":
                env.session.core.vllm_config.model_config.revision = "other"
            else:
                env.ropes["layer0.attn"].cos_sin_cache[12] *= 0.9
            with self.assertRaises(ValueError):
                handle.restore(env.llm, list(range(13)), [], 3, 5)
            self.assertFalse(env.pool.allocated)
            handle.release()

    def environment(self, *, rotary_dim=4, neox=True):
        import torch

        class FakeRope:
            head_size = 4
            is_neox_style = neox

            def __init__(self):
                self.rotary_dim = rotary_dim
                phase = (
                    torch.arange(128)[:, None]
                    / torch.arange(1, rotary_dim // 2 + 1)[None, :]
                )
                self.cos_sin_cache = torch.cat((phase.cos(), phase.sin()), -1)

            def forward_native(self):
                pass

        class Page:
            def __init__(self, block_id):
                self.block_id, self.ref_cnt = block_id, 0

        class Pool:
            def __init__(self):
                self.pages = [Page(i) for i in range(32)]
                self.source = [self.pages[i] for i in (7, 1, 9, 3)]
                self.aliases = {
                    ("source-salt", min((i + 1) * 4, 13)): b
                    for i, b in enumerate(self.source)
                }
                self.lookups = 0
                self.allocated = []
                self.fail_allocate = False

            def get_cached_block(self, key, groups):
                self.lookups += 1
                return [self.aliases[key]] if key in self.aliases else None

            def touch(self, pages):
                for page in pages:
                    page.ref_cnt += 1

            def get_new_blocks(self, count):
                if self.fail_allocate:
                    raise ValueError("no free pages")
                self.allocated = [self.pages[i * 2] for i in range(count)]
                for page in self.allocated:
                    assert page.ref_cnt == 0
                    self._maybe_evict_cached_block(page)
                    page.ref_cnt = 1
                return self.allocated

            def free_blocks(self, pages):
                for page in pages:
                    page.ref_cnt -= 1
                    assert page.ref_cnt >= 0

            def _maybe_evict_cached_block(self, page):
                self.aliases = {
                    key: value
                    for key, value in self.aliases.items()
                    if value is not page
                }

        env = SimpleNamespace(
            pool=Pool(), events=[], fail_forward=False, fail_publish=False
        )
        ropes = {name: FakeRope() for name in ("layer0.attn", "layer1.attn")}
        context = {
            name: SimpleNamespace(
                kv_cache=torch.full((32, 2, 4, 8), float("nan")),
                num_kv_heads=2,
                attn_backend=SimpleNamespace(get_name=lambda: "TRITON_ATTN"),
            )
            for name in ropes
        }
        generator = torch.Generator().manual_seed(37)
        env.raw = {name: torch.randn(13, 2, 8, generator=generator) for name in ropes}

        def rotate(keys, positions, rope):
            cos, sin = rope.cos_sin_cache[positions].chunk(2, -1)
            rotary, tail = keys[..., :rotary_dim], keys[..., rotary_dim:]
            left, right = (
                rotary.chunk(2, -1) if neox else (rotary[..., ::2], rotary[..., 1::2])
            )
            x, y = (
                left * cos[:, None] - right * sin[:, None],
                right * cos[:, None] + left * sin[:, None],
            )
            moved = (
                torch.cat((x, y), -1) if neox else torch.stack((x, y), -1).flatten(-2)
            )
            return torch.cat((moved, tail), -1)

        env.rotate = rotate
        env.source_states = {}
        for name, raw in env.raw.items():
            rows = raw.clone()
            rows[..., :4] = rotate(raw[..., :4], torch.arange(13), ropes[name])
            env.source_states[name] = rows.clone()
            cache = context[name].kv_cache.transpose(1, 2)
            for pos in range(13):
                cache[env.pool.source[pos // 4].block_id, pos % 4] = rows[pos]
        hf = SimpleNamespace(
            _commit_hash="revision", to_dict=lambda: {"model_type": "synthetic"}
        )
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                model="synthetic",
                revision="revision",
                hf_config=hf,
                dtype=torch.float32,
                max_model_len=64,
                get_vocab_size=lambda: 1000,
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            compilation_config=SimpleNamespace(static_forward_context=context),
        )
        model = SimpleNamespace(
            get_submodule=lambda name: SimpleNamespace(rotary_emb=ropes[name + ".attn"])
        )
        runner = SimpleNamespace(
            device="cpu",
            vllm_config=config,
            get_model=lambda: model,
            attn_groups=[[SimpleNamespace(layer_names=list(context))]],
        )
        editor = SimpleNamespace(
            block_size=4,
            pool=env.pool,
            salt_prefix="restored-",
            _request=lambda ids, salt: SimpleNamespace(
                ids=ids, salt=salt, num_tokens=len(ids)
            ),
            _hash_at=lambda request, end: (request.salt, end),
        )

        def publish(request, end, page):
            env.pool.aliases[request.salt, end] = page
            env.events.append(("publish", end))
            if env.fail_publish:
                raise RuntimeError("publication failed")

        editor._publish = publish
        worker = SimpleNamespace(
            model_runner=runner, synchronize_device=lambda: env.events.append(("sync",))
        )
        core = SimpleNamespace(
            scheduler=SimpleNamespace(
                kv_cache_manager=SimpleNamespace(editor=editor),
                has_unfinished_requests=lambda: False,
            ),
            vllm_config=config,
            model_executor=SimpleNamespace(driver_worker=worker),
        )
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(engine_core=SimpleNamespace(engine_core=core))
        )

        def prefill(runner, ids, start, pages, size):
            if not ids:
                return
            env.events.append(("prefill", start, list(ids)))
            if env.fail_forward:
                raise RuntimeError("forward failed")
            for layer in context.values():
                cache = layer.kv_cache.transpose(1, 2)
                for pos in range(start):
                    self.assertTrue(
                        torch.isfinite(cache[pages[pos // size], pos % size]).all()
                    )
                for pos in range(start, start + len(ids)):
                    cache[pages[pos // size], pos % size] = 1000 + pos

        class CustomOp:
            @classmethod
            def register(cls, name):
                return lambda implementation: implementation

        namespace = dict(
            torch=torch,
            CustomOp=CustomOp,
            hashlib=hashlib,
            json=json,
            RotaryEmbedding=FakeRope,
            CheckpointLayer=c.CheckpointLayer,
            checkpoint_copy_ranges=c.checkpoint_copy_ranges,
            plan_suffix_repair=planner.plan_suffix_repair,
            _prefill=prefill,
        )
        load_definitions(
            "vllm/model_executor/layers/rotary_embedding/common.py",
            ["ApplyRotaryEmb"],
            namespace,
        )
        load_definitions(
            "vllm/v1/worker/kv_cache_edit.py", ["_relocate_keys"], namespace
        )
        load_definitions(
            "vllm/v1/worker/patchkv_checkpoint.py",
            ["checkpoint_layers", "gather_checkpoint_rows", "restore_checkpoint_kv"],
            namespace,
        )
        namespace.update(
            RLock=RLock,
            uuid4=uuid4,
            asdict=asdict,
            set_current_vllm_config=lambda config: nullcontext(),
            CPUCheckpointStore=c.CPUCheckpointStore,
            CheckpointMetadata=c.CheckpointMetadata,
            CheckpointBlock=c.CheckpointBlock,
            AttentionMetadata=a.AttentionMetadata,
            plan_profiled_suffix_repair=planner.plan_profiled_suffix_repair,
            plan_attention_guided_suffix_repair=planner.plan_attention_guided_suffix_repair,
        )
        load_definitions(
            "vllm/entrypoints/patchkv.py",
            ["KVCheckpointHandle", "KVCheckpointSession"],
            namespace,
        )
        env.session = namespace["KVCheckpointSession"](llm, 1 << 20)
        env.context, env.ropes, env.runner, env.editor, env.llm = (
            context,
            ropes,
            runner,
            editor,
            llm,
        )
        return env

    def test_restore_after_complete_eviction_rope_values_and_causal_repairs(self):
        import torch

        for rotary_dim, neox in ((4, True), (2, True), (4, False)):
            for start, end, replacement, selected in (
                (3, 3, [90, 91], [2]),
                (3, 7, [], [1, 2]),
                (3, 5, [90, 91], []),
                (5, 5, [], []),
            ):
                env = self.environment(rotary_dim=rotary_dim, neox=neox)
                handle = env.session.create(
                    list(range(13)), cache_salt="source-salt", model_key="model"
                )
                self.assertTrue(all(page.ref_cnt == 0 for page in env.pool.source))
                env.pool.aliases.clear()
                for layer in env.context.values():
                    layer.kv_cache.fill_(float("nan"))
                lookups = env.pool.lookups
                salt = handle.restore(
                    env.llm,
                    list(range(13)),
                    replacement,
                    start,
                    end,
                    repair_suffix_blocks=selected,
                )
                self.assertEqual(env.pool.lookups, lookups)
                plan = planner.plan_suffix_repair(
                    13, start, end, len(replacement), 4, selected
                )
                copies = c.checkpoint_copy_ranges(
                    13, start, end, len(replacement), plan
                )
                for name, layer in env.context.items():
                    actual = torch.cat(
                        [
                            layer.kv_cache[page.block_id].transpose(0, 1)
                            for page in env.pool.allocated
                        ]
                    )[: plan.new_length]
                    self.assertTrue(torch.isfinite(actual).all())
                    for mapping in copies:
                        old = torch.arange(
                            mapping.source_start, mapping.source_start + mapping.count
                        )
                        new = torch.arange(
                            mapping.destination_start,
                            mapping.destination_start + mapping.count,
                        )
                        torch.testing.assert_close(
                            actual[new, :, 4:],
                            env.raw[name][old, :, 4:],
                            atol=0,
                            rtol=0,
                        )
                        expected = env.rotate(
                            env.raw[name][old, :, :4], new, env.ropes[name]
                        )
                        torch.testing.assert_close(
                            actual[new, :, :4], expected, atol=1e-6, rtol=1e-5
                        )
                    self.assertTrue(
                        all(page.ref_cnt == 0 for page in env.pool.allocated)
                    )
                self.assertTrue(any(key[0] == salt for key in env.pool.aliases))
                # CPU payload remains byte-identical after shifted key restoration.
                with env.session.store.lease(handle.checkpoint_id) as reader:
                    for name in env.context:
                        torch.testing.assert_close(
                            reader.read(name, 0, 13),
                            env.source_states[name],
                            atol=0,
                            rtol=0,
                        )
                handle.release()
                self.assertEqual(env.session.store.used_bytes, 0)

    def test_failure_cleanup_and_retry_allocation_forward_publication(self):
        for failure in ("fail_allocate", "fail_forward", "fail_publish"):
            env = self.environment()
            handle = env.session.create(
                list(range(13)), cache_salt="source-salt", model_key="model"
            )
            env.pool.aliases.clear()
            target = env.pool if failure == "fail_allocate" else env
            setattr(target, failure, True)
            with self.assertRaises((ValueError, RuntimeError)):
                handle.restore(
                    env.llm, list(range(13)), [90], 3, 5, repair_suffix_blocks=[2]
                )
            self.assertFalse(env.pool.aliases)
            self.assertTrue(all(page.ref_cnt == 0 for page in env.pool.allocated))
            self.assertEqual(handle.metadata.original_ids, tuple(range(13)))
            setattr(target, failure, False)
            handle.restore(
                env.llm, list(range(13)), [90], 3, 5, repair_suffix_blocks=[2]
            )
            handle.release()

    def test_full_direct_empty_chained_and_predicted_attention_recovery(self):
        env = self.environment()
        layer = a.AttentionLayerSpec("sample", 4, 2, 4, 0.5, "rope")
        attention = a.AttentionMetadata(
            "synthetic/model@revision:fp16",
            "source-salt",
            a.token_digest(range(13)),
            13,
            4,
            (12,),
            (layer,),
            tuple([1 / 13] * 13),
        )
        handle = env.session.create(
            list(range(13)),
            cache_salt="source-salt",
            model_key=attention.model_key,
            attention=attention,
        )
        profile, _, _ = profile_tests.DriftPredictorTest().fitted()
        profile = replace(
            profile,
            coefficients=(1.0, 0, 0, 0, 0, 0, 0),
            residual_margin=0,
            length_bounds=((0, 13), (0, 4), (0, 13)),
        )
        for options in (
            {},
            dict(recompute_suffix=True),
            dict(drift_profile=profile),
            dict(
                drift_profile=profile, attention_budget=a.AttentionBudget(max_tokens=2)
            ),
        ):
            env.events.clear()
            salt = handle.restore(env.llm, list(range(13)), [90], 3, 5, **options)
            self.assertTrue(any(key[0] == salt for key in env.pool.aliases))
            publish = next(
                i for i, event in enumerate(env.events) if event[0] == "publish"
            )
            self.assertEqual(env.events[publish - 1], ("sync",))
        # A restored prompt can itself be checkpointed and edited without its ancestor.
        second_ids = list(range(3)) + [90] + list(range(5, 13))
        second = env.session.create(
            second_ids, cache_salt=salt, model_key=attention.model_key
        )
        handle.release()
        env.pool.aliases.clear()
        second.restore(env.llm, second_ids, [], 0, len(second_ids))
        self.assertFalse(env.pool.allocated)
        second.release()

    def test_api_forwarding_ownership_and_incompatible_input(self):
        env = self.environment()
        env.llm._patchkv_checkpoint_session = env.session
        create = load_method("vllm/entrypoints/llm.py", "LLM", "create_kv_checkpoint")
        handle = create(
            env.llm, list(range(13)), cache_salt="source-salt", model_key="model"
        )
        with self.assertRaises(ValueError):
            create(env.llm, list(range(13)), model_key="model", max_cpu_bytes=1)
        method = load_method("vllm/entrypoints/llm.py", "LLM", "edit_kv_cache")
        env.llm.get_tokenizer = lambda: SimpleNamespace(
            encode=lambda *args, **kwargs: [90]
        )
        result = method(
            env.llm,
            list(range(13)),
            "new",
            3,
            5,
            checkpoint=handle,
            repair_suffix_blocks=[2],
        )
        self.assertEqual(
            result["prompt_token_ids"], list(range(3)) + [90] + list(range(5, 13))
        )
        for kwargs in (
            dict(cache_salt="wrong"),
            dict(recompute_suffix=True, repair_suffix_blocks=[]),
        ):
            with self.assertRaises(ValueError):
                handle.restore(env.llm, list(range(13)), [], 3, 5, **kwargs)
        with self.assertRaises(ValueError):
            handle.restore(object(), list(range(13)), [], 3, 5)
        with self.assertRaises(ValueError):
            handle.restore(env.llm, list(range(12)) + [90], [], 3, 5)
        env.session.core.scheduler.has_unfinished_requests = lambda: True
        with self.assertRaises(RuntimeError):
            handle.restore(env.llm, list(range(13)), [], 3, 5)
        handle.release()


if __name__ == "__main__":
    unittest.main()
