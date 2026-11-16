# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic CPU attention references, selection, and actual hook control flow."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from suffix_repair_test import test_patchkv_execution as execution_tests
from suffix_repair_test import test_patchkv_profile as profile_tests
from suffix_repair_test.patchkv_profile_tools import load_attention_module
from suffix_repair_test.test_patchkv_execution import load_definitions
from suffix_repair_test.test_patchkv_planner import planner

a = load_attention_module()


def spec(name="layer", hq=4, hkv=2, dim=4, scale=0.5):
    return a.AttentionLayerSpec(name, hq, hkv, dim, scale, "synthetic-rope")


def metadata(scores=None, count=16, model_key="synthetic/model@revision:fp16", size=4):
    return a.AttentionMetadata(
        model_key,
        "source-salt",
        a.token_digest(range(count)),
        count,
        size,
        (count - 1,),
        (spec(),),
        tuple(scores) if scores is not None else tuple([1.0 / count] * count),
    )


class AttentionSelectionTest(unittest.TestCase):
    def test_selected_plan_executes_through_phase1_editor_and_worker(self):
        profile, _, _ = profile_tests.DriftPredictorTest().fitted()
        profile = replace(
            profile,
            coefficients=(1.0, 0, 0, 0, 0, 0, 0),
            residual_margin=0,
            length_bounds=((0, 26), (0, 4), (0, 26)),
        )
        scores = [0.0] * 26
        scores[12], scores[24] = 0.4, 0.6
        data = metadata(scores, count=26)
        for end, replacement in ((7, [90, 91]), (5, [90, 91]), (9, [])):
            plan = planner.plan_attention_guided_suffix_repair(
                list(range(26)),
                5,
                end,
                len(replacement),
                4,
                profile,
                data,
                a.AttentionBudget(max_tokens=2),
                model_key=data.model_key,
                checkpoint_id=data.checkpoint_id,
            )
            editor = execution_tests.PatchKVEditorTest().make_editor()
            writes = []
            editor.edit(
                list(range(26)),
                replacement,
                5,
                end,
                data.checkpoint_id,
                writes.append,
                repair_suffix_blocks=list(plan.repair_blocks),
            )
            self.assertEqual(writes[0].recovery_plan, plan)
            execution_tests.PatchKVWorkerTest().check_execution(
                5, end, replacement, list(plan.repair_blocks)
            )

    def test_ties_and_dirty_exclusion_before_budgeting(self):
        data = metadata()
        self.assertEqual(
            data.select_suffix_offsets(3, 4, a.AttentionBudget(max_tokens=3)), (4, 5, 6)
        )
        self.assertEqual(
            data.select_suffix_offsets(3, 4, a.AttentionBudget(fraction=0.25)),
            (4, 5, 6),
        )
        self.assertEqual(
            data.select_suffix_offsets(3, 13, a.AttentionBudget(fraction=1)), ()
        )
        self.assertEqual(
            data.select_suffix_offsets(3, 0, a.AttentionBudget(max_tokens=0)), ()
        )

    def test_ranking_outside_dirty_window_and_budget_caps(self):
        scores = [0.0] * 16
        scores[5], scores[10], scores[15] = 0.6, 0.3, 0.1
        data = metadata(scores)
        self.assertEqual(
            data.select_suffix_offsets(3, 3, a.AttentionBudget(max_tokens=2)), (7, 12)
        )
        self.assertEqual(
            len(data.select_suffix_offsets(3, 3, a.AttentionBudget(max_tokens=100))), 10
        )

    def test_invalid_budgets_and_boundaries(self):
        for kwargs in (
            {},
            dict(max_tokens=1, fraction=0.1),
            dict(max_tokens=True),
            dict(max_tokens=-1),
            dict(fraction=float("nan")),
            dict(fraction=1.1),
        ):
            with self.assertRaises(ValueError):
                a.AttentionBudget(**kwargs)
        for suffix, dirty in ((-1, 0), (17, 0), (10, 7), (0, -1), (True, 1)):
            with self.assertRaises(ValueError):
                metadata().select_suffix_offsets(
                    suffix, dirty, a.AttentionBudget(max_tokens=1)
                )

    def test_serialization_and_compatibility(self):
        data = metadata()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.json"
            data.save(path)
            self.assertEqual(data, a.AttentionMetadata.load(path))
            payload = json.loads(path.read_text())
            for changes in (
                dict(schema_version=2),
                dict(scores=[0] * 16),
                dict(scores=[float("nan")] * 16),
                dict(query_positions=[4, 4]),
                dict(scoring="unknown"),
            ):
                path.write_text(json.dumps({**payload, **changes}))
                with self.assertRaises(ValueError):
                    a.AttentionMetadata.load(path)
        kwargs = dict(
            model_key=data.model_key,
            block_size=4,
            checkpoint_id="source-salt",
            layers=data.layers,
            query_positions=data.query_positions,
        )
        data.validate_context(list(range(16)), **kwargs)
        for changes in (
            dict(model_key="other"),
            dict(block_size=8),
            dict(checkpoint_id="other"),
            dict(query_positions=(3,)),
            dict(layers=(replace(spec(), scale=0.25),)),
            dict(layers=(replace(spec(), rope_signature="other-rope"),)),
        ):
            with self.assertRaises(ValueError):
                data.validate_context(list(range(16)), **{**kwargs, **changes})
        with self.assertRaises(ValueError):
            data.validate_context(list(range(15)) + [90], **kwargs)

    def test_drift_attention_union_shift_mapping_and_rounding(self):
        profile, _, _ = profile_tests.DriftPredictorTest().fitted()
        for length, start, end, replacement in (
            (44, 10, 12, 2),
            (42, 10, 10, 2),
            (46, 10, 14, 0),
        ):
            scores = [0.0] * length
            scores[-1] = 1
            data = metadata(scores, count=length)
            dirty = profile.predict(start, replacement, length - end)
            budget = a.AttentionBudget(max_tokens=2)
            offsets = data.select_suffix_offsets(end, dirty, budget)
            suffix_start = start + replacement
            expected = {p // 4 for p in range(suffix_start, suffix_start + dirty)}
            expected.update((suffix_start + i) // 4 for i in offsets)
            actual = planner.plan_attention_guided_suffix_repair(
                list(range(length)),
                start,
                end,
                replacement,
                4,
                profile,
                data,
                budget,
                model_key=data.model_key,
                checkpoint_id=data.checkpoint_id,
            )
            self.assertEqual(set(actual.repair_blocks), expected)
            self.assertEqual(
                set(actual.repair_blocks) & set(actual.reuse_blocks), set()
            )

    def test_attention_in_dirty_boundary_page_still_consumes_token_budget(self):
        profile, _, _ = profile_tests.DriftPredictorTest().fitted()
        # Force an in-domain 1-token dirty window without mocking the planner.
        profile = replace(
            profile, coefficients=(1.0, 0, 0, 0, 0, 0, 0), residual_margin=0
        )
        scores = [0.0] * 44
        scores[13], scores[43] = 0.8, 0.2
        data = metadata(scores, count=44)
        plan = planner.plan_attention_guided_suffix_repair(
            list(range(44)),
            10,
            12,
            2,
            4,
            profile,
            data,
            a.AttentionBudget(max_tokens=1),
            model_key=data.model_key,
            checkpoint_id=data.checkpoint_id,
        )
        self.assertEqual(plan.repair_blocks, (3,))
        # The top candidate at new position 13 is outside D even though its
        # block already intersects D. Do not silently replace it with token 43.
        self.assertEqual(
            data.select_suffix_offsets(12, 1, a.AttentionBudget(max_tokens=1)), (1,)
        )


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch unavailable")
class AttentionNumericsTest(unittest.TestCase):
    def dense(self, q, k, positions, scale):
        import torch

        mapping = torch.arange(q.shape[1]) // (q.shape[1] // k.shape[1])
        logits = (
            torch.einsum("qhd,khd->qhk", q.double(), k[:, mapping].double()) * scale
        )
        mask = torch.arange(len(k))[None, :] > torch.tensor(positions)[:, None]
        return logits.masked_fill(mask[:, None, :], -torch.inf).softmax(-1).mean((0, 1))

    def test_streaming_matches_dense_mha_gqa_mqa_and_scale(self):
        import torch

        generator = torch.Generator().manual_seed(7)
        for hq, hkv in ((4, 4), (6, 2), (4, 1)):
            q = torch.randn(4, hq, 8, generator=generator)
            k = torch.randn(19, hkv, 8, generator=generator)
            positions = (0, 3, 12, 18)
            for scale in (0.125, 0.7, 2.0):
                expected = self.dense(q, k, positions, scale)
                for kt, qt, ht in ((1, 1, 1), (5, 2, 3), (64, 8, 8)):
                    actual = a.streaming_attention_mass(
                        q,
                        k,
                        positions,
                        scale=scale,
                        key_tile=kt,
                        query_tile=qt,
                        head_tile=ht,
                    )
                    torch.testing.assert_close(
                        actual.double(), expected, atol=2e-7, rtol=2e-6
                    )

    def test_causal_mask_probe_denominator_and_extreme_logits(self):
        import torch

        q = torch.ones(2, 2, 2)
        k = torch.ones(8, 1, 2)
        k[4:] = 10000
        mass = a.streaming_attention_mass(q, k, (1, 3), scale=1, key_tile=2)
        torch.testing.assert_close(
            mass, torch.tensor([0.375, 0.375, 0.125, 0.125, 0, 0, 0, 0])
        )
        # A generated probe key participates in softmax, not just old tokens.
        mass = a.streaming_attention_mass(q[:1], k, (7,), scale=1, key_tile=2)
        self.assertAlmostEqual(mass.sum().item(), 1)
        self.assertAlmostEqual(mass[:4].sum().item(), 0)

    def test_logits_allocations_are_tiled_not_full_attention(self):
        import torch

        q, k = torch.ones(5, 12, 4), torch.ones(27, 3, 4)
        original = torch.einsum
        shapes = []

        def record(equation, query, keys):
            output = original(equation, query, keys)
            shapes.append(tuple(output.shape))
            return output

        with patch.object(torch, "einsum", side_effect=record):
            a.streaming_attention_mass(
                q,
                k,
                (0, 3, 8, 15, 26),
                scale=0.5,
                key_tile=4,
                query_tile=2,
                head_tile=3,
            )
        self.assertTrue(shapes)
        self.assertTrue(all(q <= 2 and h <= 3 and k <= 4 for q, h, k in shapes))

    def test_accumulator_chunking_layers_probes_and_normalization(self):
        import torch

        generator = torch.Generator().manual_seed(9)
        specs = (spec("a"), spec("b", hq=2, hkv=1, scale=0.2))
        accumulator = a.AttentionAccumulator(
            range(7),
            model_key="model",
            checkpoint_id="salt",
            block_size=4,
            query_positions=(2, 8),
            layers=specs,
            key_tile=3,
        )
        expected = torch.zeros(7, dtype=torch.float64)
        for layer in specs:
            q = torch.randn(9, layer.num_query_heads, 4, generator=generator)
            k = torch.randn(9, layer.num_kv_heads, 4, generator=generator)
            for lo, hi in ((0, 4), (4, 7), (7, 9)):
                accumulator.record(layer.name, torch.arange(lo, hi), q[lo:hi], k[lo:hi])
            expected += self.dense(q[[2, 8]], k, (2, 8), layer.scale)[:7]
        data = accumulator.finalize()
        torch.testing.assert_close(
            torch.tensor(data.scores, dtype=torch.float64),
            expected / expected.sum(),
            atol=2e-7,
            rtol=2e-6,
        )

    def test_invalid_inputs_and_incomplete_capture(self):
        import torch

        q, k = torch.ones(2, 4, 4), torch.ones(4, 2, 4)
        for positions, changes in (
            ((0, 4), {}),
            ((0, 0), {}),
            ((-1, 1), {}),
            ((0, 1), dict(scale=float("nan"))),
            ((0, 1), dict(key_tile=0)),
        ):
            with self.assertRaises(ValueError):
                a.streaming_attention_mass(
                    q, k, positions, **{**dict(scale=1), **changes}
                )
        with self.assertRaises(ValueError):
            a.streaming_attention_mass(
                q, torch.full_like(k, float("nan")), (0, 1), scale=1
            )
        accumulator = a.AttentionAccumulator(
            range(4),
            model_key="m",
            checkpoint_id="s",
            block_size=4,
            query_positions=(3,),
            layers=(spec(),),
        )
        accumulator.record("layer", torch.arange(2), q, k[:2])
        with self.assertRaises(ValueError):
            accumulator.finalize()
        with self.assertRaises(ValueError):
            accumulator.record("layer", torch.arange(2), q, k[:2])


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch unavailable")
class AttentionCaptureTest(unittest.TestCase):
    def setup_capture(self, *, rotary_dim=4, neox=True, positions=(2, 6)):
        import torch

        class FakeRope(torch.nn.Module):
            head_size = 4
            is_neox_style = neox

            def __init__(self):
                super().__init__()
                self.rotary_dim = rotary_dim
                phase = (
                    torch.arange(64)[:, None]
                    / torch.arange(1, rotary_dim // 2 + 1)[None, :]
                )
                self.cos_sin_cache = torch.cat((phase.cos(), phase.sin()), -1)

            def forward_native(self, positions, q, k):
                cos, sin = self.cos_sin_cache[positions].chunk(2, -1)

                def rotate(tensor):
                    rotary, tail = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
                    left, right = (
                        rotary.chunk(2, -1)
                        if neox
                        else (rotary[..., ::2], rotary[..., 1::2])
                    )
                    x = left * cos[:, None, :] - right * sin[:, None, :]
                    y = right * cos[:, None, :] + left * sin[:, None, :]
                    rotated = (
                        torch.cat((x, y), -1)
                        if neox
                        else torch.stack((x, y), -1).flatten(-2)
                    )
                    return torch.cat((rotated, tail), -1)

                return rotate(q), rotate(k)

            def forward(self, positions, q, k):
                return self.forward_native(positions, q, k)

        class ToyAttention(torch.nn.Module):
            num_heads = 4
            num_kv_heads = 2
            head_size = 4
            kv_cache_dtype = "auto"
            attn_type = "decoder"
            sliding_window = None
            attn_backend = SimpleNamespace(get_name=lambda: "TRITON_ATTN")

            def __init__(self):
                super().__init__()
                self.impl = SimpleNamespace(scale=0.37)
                self.queries = []
                self.keys = []

            def forward(self, query, key, value):
                self.queries.append(query.reshape(-1, 4, 4).clone())
                self.keys.append(key.reshape(-1, 2, 4).clone())
                return query.clone()

        class Qwen2Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rotary_emb = FakeRope()
                self.attn = ToyAttention()

            def forward(self, positions, ids):
                query = torch.sin(
                    ids[:, None, None].float() + torch.arange(16).reshape(1, 4, 4) * 0.3
                )
                keys = torch.cos(
                    ids[:, None, None].float() + torch.arange(8).reshape(1, 2, 4) * 0.4
                )
                query, keys = self.rotary_emb(positions, query, keys)
                return self.attn(query.flatten(1), keys.flatten(1), keys.flatten(1))

        # Exercise the production adapter's restricted parent-class contract.
        Qwen2Attention.__module__ = "vllm.model_executor.models.qwen2"

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = Qwen2Attention()

            def forward(self, input_ids, positions):
                return self.layer0(positions, input_ids)

        model = ToyModel()
        config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=True),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            speculative_config=None,
            lora_config=None,
            kv_transfer_config=None,
            compilation_config=SimpleNamespace(
                static_forward_context={"layer0.attn": model.layer0.attn}
            ),
        )
        runner = SimpleNamespace(
            get_model=lambda: model,
            vllm_config=config,
            kv_cache_config=SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4))
                ]
            ),
        )
        namespace = dict(
            torch=torch,
            hashlib=hashlib,
            json=json,
            partial=partial,
            RotaryEmbedding=FakeRope,
            AttentionAccumulator=a.AttentionAccumulator,
            AttentionLayerSpec=a.AttentionLayerSpec,
            AttentionMetadata=a.AttentionMetadata,
        )
        load_definitions(
            "vllm/v1/worker/patchkv_attention.py",
            ["CheckpointAttentionCollector"],
            namespace,
        )
        cls = namespace["CheckpointAttentionCollector"]
        kwargs = dict(
            model_key="toy@revision:fp32",
            checkpoint_id="source",
            block_size=4,
            layer_names=("layer0.attn",),
            query_positions=positions,
        )
        return runner, model, cls, kwargs

    def test_post_rope_hook_capture_full_partial_and_interleaved(self):
        import torch

        for rotary_dim, neox in ((4, True), (2, True), (4, False), (2, False)):
            runner, model, cls, kwargs = self.setup_capture(
                rotary_dim=rotary_dim, neox=neox
            )
            chunks = ((0, 2), (2, 5), (5, 7))
            baseline = [
                model(input_ids=torch.arange(lo, hi), positions=torch.arange(lo, hi))
                for lo, hi in chunks
            ]
            model.layer0.attn.queries.clear()
            model.layer0.attn.keys.clear()
            capture = cls(runner, list(range(5)), **kwargs)
            with capture:
                actual = [
                    model(
                        input_ids=torch.arange(lo, hi), positions=torch.arange(lo, hi)
                    )
                    for lo, hi in chunks
                ]
            data = capture.finalize()
            q, k = (
                torch.cat(model.layer0.attn.queries),
                torch.cat(model.layer0.attn.keys),
            )
            expected = AttentionNumericsTest().dense(q[[2, 6]], k, (2, 6), 0.37)[:5]
            expected /= expected.sum()
            torch.testing.assert_close(
                torch.tensor(data.scores).double(), expected, atol=2e-7, rtol=2e-6
            )
            for x, y in zip(actual, baseline):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            self.assertEqual(len(model._forward_pre_hooks), 0)
            self.assertEqual(len(model.layer0.attn._forward_pre_hooks), 0)
            self.assertFalse(capture.accumulator.keys)
            self.assertIs(data, capture.finalize())
            with self.assertRaises(ValueError):
                capture.__enter__()

    def test_hooks_removed_on_bad_tokens_cached_pass_and_incomplete_queries(self):
        import torch

        for ids, positions in (([0, 90], [0, 1]), ([4], [4])):
            runner, model, cls, kwargs = self.setup_capture()
            capture = cls(runner, list(range(5)), **kwargs)
            with self.assertRaises(ValueError), capture:
                model(input_ids=torch.tensor(ids), positions=torch.tensor(positions))
            self.assertEqual(len(model._forward_pre_hooks), 0)
            self.assertEqual(len(model.layer0.attn._forward_pre_hooks), 0)
            with self.assertRaises(ValueError):
                capture.finalize()
        runner, model, cls, kwargs = self.setup_capture()
        capture = cls(runner, list(range(5)), **kwargs)
        with capture:
            model(input_ids=torch.arange(5), positions=torch.arange(5))
        with self.assertRaises(ValueError):
            capture.finalize()

    def test_adapter_rejects_unsupported_attention_and_parallel_modes(self):
        mutations = (
            lambda runner, model: setattr(model.layer0.attn.impl, "alibi_slopes", [1]),
            lambda runner, model: setattr(model.layer0.attn.impl, "logits_soft_cap", 3),
            lambda runner, model: setattr(model.layer0.attn, "sliding_window", 16),
            lambda runner, model: setattr(model.layer0.attn, "kv_cache_dtype", "fp8"),
            lambda runner, model: setattr(
                runner.vllm_config.model_config, "enforce_eager", False
            ),
            lambda runner, model: setattr(
                runner.vllm_config.parallel_config, "tensor_parallel_size", 2
            ),
        )
        for mutate in mutations:
            runner, model, cls, kwargs = self.setup_capture()
            mutate(runner, model)
            with self.assertRaises(ValueError):
                cls(runner, list(range(5)), **kwargs)
            self.assertFalse(model._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
