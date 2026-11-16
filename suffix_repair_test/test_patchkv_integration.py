# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real metadata/prefill/checkpoint functions with a tiny CPU causal paged model."""

import importlib.util
import unittest
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

from suffix_repair_test import test_patchkv_checkpoint as checkpoint_tests
from suffix_repair_test import test_patchkv_execution as execution_tests
from suffix_repair_test import test_patchkv_quantization as quantization_tests
from suffix_repair_test.test_patchkv_execution import load_definitions, load_method


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch unavailable")
class PatchKVIntegrationTest(unittest.TestCase):
    def test_resident_direct_full_and_manual_use_real_prefill(self):
        import torch

        for options in (
            {},
            dict(recompute_suffix=True),
            dict(repair_suffix_blocks=[2]),
        ):
            env = self.environment()
            cls = execution_tests.EDITOR_NAMESPACE["KVCacheEditor"]
            editor = cls.__new__(cls)
            vars(editor).update(vars(env.editor))
            source_pages = [page.block_id for page in env.pool.source]
            before = self.states(env, source_pages, 13)
            edited_ids = env.ids[:5] + [90] + env.ids[7:]
            env.prefill(env.runner, edited_ids, 0, torch.tensor([10, 11, 12]), 4)
            fresh = self.states(env, [10, 11, 12], len(edited_ids))
            edits = []

            def write(edit, edits=edits, env=env):
                edits.append(edit)
                env.namespace["write_kv_cache_edit"](env.runner, edit)

            editor.edit(env.ids, [90], 5, 7, "source-salt", write, **options)
            actual = self.states(env, edits[0].destination, len(edited_ids))
            for name in actual:
                self.assertTrue(torch.isfinite(actual[name]).all())
                torch.testing.assert_close(
                    self.states(env, source_pages, 13)[name],
                    before[name],
                    atol=0,
                    rtol=0,
                )
                if options.get("recompute_suffix"):
                    torch.testing.assert_close(
                        actual[name], fresh[name], atol=0.003, rtol=0.003
                    )

    def environment(self):
        import torch

        env = quantization_tests.QuantizedCheckpointTest().environment()
        namespace = env.session.restore.__func__.__globals__
        namespace.update(
            dataclass=dataclass, field=field, Mapping=Mapping, replace=replace
        )
        load_definitions(
            "vllm/v1/attention/backend.py", ["CommonAttentionMetadata"], namespace
        )
        load_definitions("vllm/v1/worker/utils.py", ["AttentionGroup"], namespace)
        load_definitions(
            "vllm/v1/worker/gpu/attn_utils.py",
            ["build_attn_metadata", "build_slot_mappings_by_layer"],
            namespace,
        )
        common_cls = namespace["CommonAttentionMetadata"]
        group_cls = namespace["AttentionGroup"]
        names = list(env.context)
        builder = SimpleNamespace(
            supports_update_block_table=False, build=lambda prefix, common, **kw: common
        )
        group = group_cls(
            backend=object,
            layer_names=names,
            kv_cache_spec=SimpleNamespace(block_size=4),
            kv_cache_group_id=0,
            kernel_block_size=2,
            metadata_builders=[builder],
        )
        group.kernel_block_table = torch.zeros(1, 1, 64, dtype=torch.int32)
        group.kernel_block_offsets = torch.arange(2, dtype=torch.int32)
        env.runner.attn_groups = [[group]]
        env.runner.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(layer_names=names, kv_cache_spec=group.kv_cache_spec)
            ]
        )
        env.runner.vllm_config.scheduler_config = SimpleNamespace(
            max_num_batched_tokens=3
        )
        active = {}
        env.metadata_records = []

        @contextmanager
        def forward_context(metadata, config, **kwargs):
            active.update(metadata=metadata, slots=kwargs["slot_mapping"])
            env.metadata_records.append(metadata[names[0]])
            yield
            active.clear()

        original_get_submodule = env.runner.get_model().get_submodule

        class TinyPagedModel(torch.nn.Module):
            def get_submodule(self, name):
                return original_get_submodule(name)

            def forward(self, input_ids, positions):
                hidden = torch.sin(
                    input_ids[:, None].float() * torch.arange(1, 9)[None, :] / 101
                )
                for index, name in enumerate(names):
                    metadata = active["metadata"][name]
                    assert isinstance(metadata, common_cls) and metadata.causal is True
                    assert metadata.max_seq_len == int(positions[-1]) + 1
                    query = hidden.reshape(-1, 2, 4)
                    keys = env.rotate(query * 0.7, positions, env.ropes[name])
                    values = torch.cos(query + index)
                    query = env.rotate(query, positions, env.ropes[name])
                    cache = env.context[name].kv_cache.transpose(1, 2)
                    slots = active["slots"][name]
                    cache[slots // 2, slots % 2] = torch.cat((keys, values), -1).half()
                    outputs = []
                    for row, position in enumerate(positions.tolist()):
                        visible = torch.arange(position + 1)
                        pages = metadata.block_table_tensor[0, visible // 2].long()
                        context = cache[pages, visible % 2].float()
                        assert torch.isfinite(context).all(), (
                            "Unseeded causal dependency"
                        )
                        scores = (
                            torch.einsum("hd,khd->hk", query[row], context[..., :4])
                            * 0.5
                        )
                        outputs.append(
                            torch.einsum(
                                "hk,khd->hd", scores.softmax(-1), context[..., 4:]
                            )
                        )
                    hidden = hidden + torch.stack(outputs).flatten(1) * 0.1
                self.logits = hidden @ torch.cos(
                    torch.arange(8 * 32).reshape(8, 32).float()
                )
                return hidden

        model = TinyPagedModel()
        env.runner.get_model = lambda: model
        # Split the manager's 4-token pages into 2-token kernel views.
        for layer in env.context.values():
            layer.kv_cache = (
                layer.kv_cache.permute(0, 2, 1, 3)
                .contiguous()
                .reshape(64, 2, 2, 8)
                .transpose(1, 2)
            )
        namespace.update(
            set_forward_context=forward_context, cdiv=lambda n, d: (n + d - 1) // d
        )
        load_definitions(
            "vllm/v1/worker/kv_cache_edit.py",
            ["_prefill", "_copy_tokens", "write_kv_cache_edit"],
            namespace,
        )
        env.prefill = namespace["_prefill"]
        env.namespace, env.model, env.group = namespace, model, group

        # Exercise the actual worker-wrapper delegation used by UniProcExecutor.
        class Wrapper:
            __getattr__ = load_method(
                "vllm/v1/worker/worker_base.py", "WorkerWrapperBase", "__getattr__"
            )

        wrapper = Wrapper()
        wrapper.worker = env.session.worker
        env.session.core.model_executor.driver_worker = wrapper
        env.session = type(env.session)(env.llm, 1 << 20)
        env.llm._patchkv_checkpoint_session = env.session
        env.llm.get_tokenizer = lambda: SimpleNamespace(
            encode=lambda *args, **kwargs: [90]
        )
        env.ids = list(range(13))
        env.prefill(
            env.runner,
            env.ids,
            0,
            torch.tensor([b.block_id for b in env.pool.source]),
            4,
        )
        return env

    def states(self, env, pages, length):

        return {
            name: env.namespace["gather_checkpoint_rows"](
                layer.kv_cache.transpose(1, 2), pages, 4, 0, length
            ).clone()
            for name, layer in env.context.items()
        }

    def test_full_restore_matches_fresh_with_real_metadata_and_kernel_splitting(self):
        import torch

        env = self.environment()
        edited = env.ids[:3] + [90] + env.ids[5:]
        handle = env.session.create(
            env.ids, model_key="model", cache_salt="source-salt"
        )
        env.prefill(env.runner, edited, 0, torch.tensor([10, 11, 12]), 4)
        fresh = self.states(env, [10, 11, 12], len(edited))
        env.pool.aliases.clear()
        for layer in env.context.values():
            layer.kv_cache.fill_(float("nan"))
        handle.restore(env.llm, env.ids, [90], 3, 5, recompute_suffix=True)
        actual = self.states(env, [b.block_id for b in env.pool.allocated], len(edited))
        for name in actual:
            torch.testing.assert_close(
                actual[name], fresh[name], rtol=0.003, atol=0.003
            )
        # Manager IDs [0,2,4] must become [0,1,4,5,8,9], exactly once.
        table = env.metadata_records[-1].block_table_tensor
        self.assertEqual(table[0, :6].tolist(), [0, 1, 4, 5, 8, 9])
        handle.release()
        self.assertEqual(env.session.store.used_bytes, 0)

    def test_all_modes_public_api_eviction_generation_and_cleanup(self):
        import torch

        env = self.environment()
        a = checkpoint_tests.a
        importance = a.AttentionMetadata(
            "model",
            "source-salt",
            a.token_digest(env.ids),
            13,
            4,
            (12,),
            (a.AttentionLayerSpec("sample", 2, 2, 4, 0.5, "toy"),),
            tuple([1 / 13] * 13),
        )
        create = load_method("vllm/entrypoints/llm.py", "LLM", "create_kv_checkpoint")
        edit = load_method("vllm/entrypoints/llm.py", "LLM", "edit_kv_cache")
        source = create(
            env.llm,
            env.ids,
            model_key="model",
            cache_salt="source-salt",
            attention=importance,
        )
        encoded = source.compress(
            quantization_tests.p.PrecisionPolicy.experimental_fixed(
                "model", 4, "K8/V4"
            ),
            protected_prefix_tokens=3,
        )
        profile, _, _ = checkpoint_tests.profile_tests.DriftPredictorTest().fitted()
        profile = replace(
            profile,
            config=replace(profile.config, model_key="model"),
            coefficients=(1.0, 0, 0, 0, 0, 0, 0),
            residual_margin=0,
            length_bounds=((0, 13), (0, 4), (0, 13)),
        )
        for handle, options in (
            (source, {}),
            (source, dict(recompute_suffix=True)),
            (source, dict(repair_suffix_blocks=[1, 2])),
            (source, dict(drift_profile=profile)),
            (
                source,
                dict(
                    drift_profile=profile,
                    attention_budget=a.AttentionBudget(max_tokens=1),
                ),
            ),
            (
                encoded,
                dict(
                    drift_profile=profile,
                    attention_budget=a.AttentionBudget(max_tokens=1),
                ),
            ),
        ):
            env.pool.aliases.clear()
            for layer in env.context.values():
                layer.kv_cache.fill_(float("nan"))
            lookups = env.pool.lookups
            prompt = edit(env.llm, env.ids, "new", 3, 5, checkpoint=handle, **options)
            self.assertEqual(env.pool.lookups, lookups)
            pages = [b.block_id for b in env.pool.allocated] + [30]
            # Resend the last prompt token, then process one generated token.
            ids = prompt["prompt_token_ids"]
            outputs = []
            for position in (len(ids) - 1, len(ids)):
                token = ids[-1] if not outputs else outputs[-1]
                env.prefill(env.runner, [token], position, torch.tensor(pages), 4)
                outputs.append(int(env.model.logits[-1].argmax()))
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(0 <= token < 32 for token in outputs))
            stats = handle.restore_statistics
            stats["format_bytes"]["tampered"] = 1
            self.assertNotIn("tampered", handle.restore_statistics["format_bytes"])
            self.assertTrue(all(page.ref_cnt == 0 for page in env.pool.allocated))
        env.session.close()
        self.assertEqual(env.session.store.used_bytes, 0)
        with self.assertRaises(KeyError):
            _ = source.metadata

    def test_packed_stride_rejected_before_snapshot(self):
        env = self.environment()
        env.group.kernel_block_stride = 3
        with self.assertRaisesRegex(ValueError, "packed"):
            env.session.create(env.ids, model_key="model", cache_salt="source-salt")
        self.assertEqual(env.session.store.used_bytes, 0)

    def test_bound_transport_policy_rejects_mismatch_before_allocation(self):
        env = self.environment()
        p = quantization_tests.p
        env.namespace["repair_policy_fingerprint"] = p.repair_policy_fingerprint
        load_definitions(
            "vllm/v1/core/patchkv_transport.py",
            ["validate_transport_repair"],
            env.namespace,
        )
        source = env.session.create(
            env.ids, model_key="model", cache_salt="source-salt"
        )
        policy = replace(
            p.PrecisionPolicy.experimental_fixed("model", 4, "FP16"),
            repair_fingerprint=p.repair_policy_fingerprint(mode="direct"),
        )
        encoded = source.compress(policy, protected_prefix_tokens=3)
        edit = load_method("vllm/entrypoints/llm.py", "LLM", "edit_kv_cache")
        before = len(env.pool.allocated)
        with self.assertRaisesRegex(ValueError, "repair policy"):
            edit(
                env.llm,
                env.ids,
                "new",
                3,
                5,
                checkpoint=encoded,
                repair_suffix_blocks=[1],
            )
        self.assertEqual(len(env.pool.allocated), before)
        prompt = edit(env.llm, env.ids, "new", 3, 5, checkpoint=encoded)
        self.assertEqual(len(prompt["prompt_token_ids"]), 12)
        env.session.close()
        self.assertEqual(env.session.store.used_bytes, 0)

    def test_clear_refuses_active_lease_then_invalidates_handles(self):
        env = self.environment()
        handle = env.session.create(
            env.ids, model_key="model", cache_salt="source-salt"
        )
        with (
            env.session.store.lease(handle.checkpoint_id),
            self.assertRaises(RuntimeError),
        ):
            env.session.close()
        env.session.close()
        self.assertFalse(handle.release())

    def test_resident_engine_rejects_active_requests_and_invalid_tokens(self):
        method = load_method("vllm/v1/engine/core.py", "EngineCore", "edit_kv_cache")
        env = self.environment()
        core = env.session.core
        core.scheduler.has_unfinished_requests = lambda: True
        with self.assertRaises(RuntimeError):
            method(core, env.ids, [90], 3, 5, "source-salt")
        core.scheduler.has_unfinished_requests = lambda: False
        for ids, tokens, start in (
            ([1001], [90], 0),
            (env.ids, [True], 3),
            (env.ids, [90], True),
        ):
            with self.assertRaises(ValueError):
                method(
                    core, ids, tokens, start, 5 if len(ids) > 1 else 1, "source-salt"
                )


if __name__ == "__main__":
    unittest.main()
