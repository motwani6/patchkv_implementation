# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic CPU codec/policy/transport checks; no real quality calibration."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from suffix_repair_test import test_patchkv_checkpoint as checkpoint_tests
from suffix_repair_test.patchkv_profile_tools import load_transport_support
from suffix_repair_test.test_patchkv_execution import load_definitions, load_method
from suffix_repair_test.test_patchkv_planner import planner

q = load_transport_support("patchkv_quantization")
p = load_transport_support("patchkv_precision")
c, a = checkpoint_tests.c, checkpoint_tests.a


def transport_module():
    namespace = dict(
        CODEC=q.CODEC,
        encode_kv_block=q.encode_kv_block,
        value_residual=q.value_residual,
        json=json,
        asdict=asdict,
        replace=replace,
    )
    load_definitions(
        "vllm/v1/core/patchkv_transport.py",
        [
            "EncodedCheckpointPayload",
            "EncodedCheckpointReader",
            "validate_transport_prefix",
            "create_quantized_checkpoint",
        ],
        namespace,
    )
    return SimpleNamespace(**namespace)


class PrecisionPolicyTest(unittest.TestCase):
    def test_cpu_fit_cli_and_zero_quality_baseline_rejection(self):
        candidates = [self.candidate(0), self.candidate(1)]
        tasks = ("train-a", "train-b", "held-a", "held-b")
        evidence = [
            p.QualityMeasurement(
                task, policy.policy_id, 1, 1, 100, 1, "synthetic_f1", "unit-fixture"
            )
            for policy in candidates
            for task in tasks
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "measurements.json"
            path.write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        candidates=[asdict(x) for x in candidates],
                        measurements=[asdict(x) for x in evidence],
                    )
                )
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "from suffix_repair_test.patchkv_precision_cli import main; "
                        "main(); assert 'torch' not in sys.modules; "
                        "assert 'vllm' not in sys.modules"
                    ),
                    "--measurements",
                    str(path),
                    "--calibration-tasks",
                    *tasks[:2],
                    "--validation-tasks",
                    *tasks[2:],
                    "--max-quality-loss",
                    "0",
                    "--output-dir",
                    str(root / "fit"),
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parents[1],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(
                p.PrecisionPolicy.load(
                    root / "fit/precision-policy.json"
                ).quality_calibrated
            )

        def zero_quality(policy, task):
            return p.QualityMeasurement(
                task, policy.policy_id, 0, 0, 100, 1, "synthetic_f1", "unit-fixture"
            )

        with self.assertRaises(ValueError):
            p.calibrate_precision_policy(
                candidates, tasks[:2], tasks[2:], zero_quality, 0
            )

    def candidate(self, g4=0.01):
        return p.PrecisionPolicy(
            "model",
            4,
            config_fingerprint="a" * 64,
            tau_low=0.01,
            tau_high=0.1,
            v4_error_limit=g4,
        )

    def test_thresholds_lowbit_gate_and_explicit_experiments(self):
        policy = self.candidate()
        self.assertEqual(policy.choose(0.2, 0), "FP16")
        self.assertEqual(policy.choose(0.1, 0), "FP16")
        self.assertEqual(policy.choose(0.01, 0), "K8/V8")
        self.assertEqual(policy.choose(0, 0.01), "K8/V4")
        self.assertEqual(policy.choose(0, 0.02), "K8/V8")
        self.assertFalse(policy.quality_calibrated)
        fixed = p.PrecisionPolicy.experimental_fixed("model", 4, "K8/V4")
        self.assertFalse(fixed.quality_calibrated)
        self.assertEqual(fixed.choose(100, 100), "K8/V4")
        with self.assertRaises(ValueError):
            replace(policy, mode="calibrated")

    def test_whole_policy_quality_selects_threshold_and_synthetic_never_calibrated(
        self,
    ):
        conservative, aggressive = self.candidate(0), self.candidate(1)
        calls = []

        def evaluate(policy, task):
            calls.append((policy.policy_id, task))
            good = policy.v4_error_limit == 0
            return p.QualityMeasurement(
                task,
                policy.policy_id,
                1,
                1 if good else 0,
                500 if good else 100,
                2,
                "synthetic_answer_exact_match",
                "unit-fixture",
                "synthetic",
            )

        result, evidence = p.calibrate_precision_policy(
            [aggressive, conservative],
            ("train-a", "train-b"),
            ("held-a", "held-b"),
            evaluate,
            max_quality_loss=0,
        )
        self.assertEqual(result.v4_error_limit, 0)
        self.assertFalse(result.quality_calibrated)
        self.assertEqual(len(evidence), 6)
        self.assertTrue(
            all(
                pid == conservative.policy_id
                for pid, task in calls
                if task.startswith("held")
            )
        )
        with self.assertRaises(ValueError):
            replace(result, mode="calibrated")

    def test_quality_heldout_failure_and_task_leakage(self):
        policy = self.candidate()

        def evaluate(rule, task):
            return p.QualityMeasurement(
                task,
                rule.policy_id,
                1,
                0 if task.startswith("held") else 1,
                100,
                1,
                "synthetic_exact",
                "unit-fixture",
            )

        with self.assertRaisesRegex(ValueError, "held-out"):
            p.calibrate_precision_policy(
                [policy], ("a", "b"), ("held-a", "held-b"), evaluate, 0
            )
        with self.assertRaises(ValueError):
            p.calibrate_precision_policy([policy], ("a", "b"), ("b", "c"), evaluate, 0)
        with self.assertRaises(ValueError):
            p.calibrate_precision_policy(
                [policy], ("held-a", "held-b"), ("c", "d"), evaluate, 0
            )

    def test_policy_roundtrip_and_configuration_binding(self):
        policy = self.candidate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            policy.save(path)
            self.assertEqual(policy, p.PrecisionPolicy.load(path))
        one = json.dumps(
            dict(
                model="m",
                layers=[dict(name="a", dtype="float16", rope_signature="one")],
            )
        )
        two = json.dumps(
            dict(
                model="m",
                layers=[dict(name="a", dtype="float16", rope_signature="two")],
            )
        )
        self.assertEqual(
            p.transport_config_fingerprint(one), p.transport_config_fingerprint(two)
        )
        data = replace(
            checkpoint_tests.metadata(), model_key="model", model_config_json=one
        )
        matched = replace(
            policy, config_fingerprint=p.transport_config_fingerprint(one)
        )
        matched.validate_context(data)
        importance = a.AttentionMetadata(
            "model",
            "source-salt",
            a.token_digest(range(13)),
            13,
            4,
            (12,),
            (a.AttentionLayerSpec("sample", 4, 2, 4, 0.5, "toy"),),
            tuple([1 / 13] * 13),
        )
        data = replace(data, attention_json=json.dumps(asdict(importance)))
        bound = replace(
            matched,
            attention_fingerprint=p.attention_config_fingerprint(
                data.attention_json, 13
            ),
        )
        bound.validate_context(data)
        altered = replace(importance, query_positions=(11,))
        with self.assertRaises(ValueError):
            bound.validate_context(
                replace(data, attention_json=json.dumps(asdict(altered)))
            )
        with self.assertRaises(ValueError):
            policy.validate_context(data)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch unavailable")
class KVCodecTest(unittest.TestCase):
    def test_int4_all_codes_odd_padding_and_noncontiguous_input(self):
        import torch

        values = torch.arange(-8, 8, dtype=torch.int8).reshape(2, 8).t()
        self.assertFalse(values.is_contiguous())
        torch.testing.assert_close(q.unpack_int4(q.pack_int4(values), 2), values)
        odd = torch.tensor([[[-8, -1, 0, 1, 7]]], dtype=torch.int8)
        packed = q.pack_int4(odd)
        self.assertEqual(packed.tolist(), [[[248, 16, 7]]])
        torch.testing.assert_close(q.unpack_int4(packed, 5), odd)
        bad = packed.clone()
        bad[..., -1] |= 16
        with self.assertRaises(ValueError):
            q.unpack_int4(bad, 5)
        with self.assertRaises(ValueError):
            q.pack_int4(torch.tensor([8], dtype=torch.int8))

    def test_error_bound_payload_sizes_and_zero_values(self):
        import torch

        generator = torch.Generator().manual_seed(51)
        for dim in (5, 16):
            source = torch.randn(7, 3, 2 * dim, generator=generator).half()
            for fmt in q.FORMATS:
                block = q.encode_kv_block(source, fmt)
                decoded = block.decode_to(dtype=torch.float32)
                if fmt == "FP16":
                    torch.testing.assert_close(decoded, source.float(), atol=0, rtol=0)
                    expected_bytes = 7 * 3 * 4 * dim
                else:
                    k_scale, v_scale = block.tensors[2:]
                    self.assertTrue(
                        (
                            (decoded[..., :dim] - source[..., :dim]).abs()
                            <= k_scale / 2 + 1e-6
                        ).all()
                    )
                    self.assertTrue(
                        (
                            (decoded[..., dim:] - source[..., dim:]).abs()
                            <= v_scale / 2 + 1e-6
                        ).all()
                    )
                    expected_bytes = (
                        7 * 3 * (dim + ((dim + 1) // 2 if fmt == "K8/V4" else dim) + 8)
                    )
                self.assertEqual(block.nbytes, expected_bytes)
                zero = q.encode_kv_block(torch.zeros_like(source), fmt)
                self.assertEqual(zero.decode_to().count_nonzero().item(), 0)
                self.assertTrue(torch.isfinite(zero.decode_to()).all())

    def test_codec_serialization_and_slice_ownership(self):
        import torch

        for fmt in q.FORMATS:
            source = torch.arange(6 * 2 * 10).reshape(6, 2, 10).half()
            block = q.encode_kv_block(source, fmt)
            expected = block.decode_to()
            source.zero_()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "block.json"
                block.save(path)
                loaded = q.EncodedKVBlock.load(path)
                torch.testing.assert_close(loaded.decode_to(), expected, atol=0, rtol=0)
                with self.assertRaises(ValueError):
                    q.EncodedKVBlock.load(path, max_bytes=1)
            selected = block.slice(1, 2)
            torch.testing.assert_close(
                selected.decode_to(), expected[1:3], atol=0, rtol=0
            )
            for tensor in selected.tensors:
                tensor.zero_()
            torch.testing.assert_close(block.decode_to(), expected, atol=0, rtol=0)

    def test_invalid_values_headers_scales_and_payload_lengths(self):
        import torch

        with self.assertRaises(ValueError):
            q.encode_kv_block(torch.full((2, 1, 8), float("nan")), "K8/V4")
        block = q.encode_kv_block(torch.ones(2, 1, 8), "K8/V8")
        data = block.to_dict()
        for changes in (dict(codec="unknown"), dict(count=3), dict(format="K4/V4")):
            with self.assertRaises(ValueError):
                q.EncodedKVBlock.from_dict({**data, **changes})
        block.tensors[2].zero_()
        with self.assertRaises(ValueError):
            block.validate()


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch unavailable")
class QuantizedCheckpointTest(unittest.TestCase):
    def test_three_frozen_formats_attention_risk_and_noncontiguous_repairs(self):
        import torch

        env = self.environment()
        importance = a.AttentionMetadata(
            "model",
            "source-salt",
            a.token_digest(range(13)),
            13,
            4,
            (12,),
            (a.AttentionLayerSpec("sample", 4, 2, 4, 0.5, "toy"),),
            tuple([1 / 13] * 13),
        )
        original = env.session.create(
            list(range(13)),
            cache_salt="source-salt",
            model_key="model",
            attention=importance,
        )
        risks = []
        with env.session.store.lease(original.checkpoint_id) as reader:
            for block in original.metadata.blocks:
                errors = []
                for layer in original.metadata.layers:
                    source = reader.read(layer.name, block.source_start, block.count)
                    errors.append(
                        q.value_residual(
                            source,
                            q.encode_kv_block(source, "K8/V8").decode_to(
                                dtype=torch.float32
                            ),
                        )
                    )
                z = torch.sqrt(torch.stack(errors).mean(0) / 13)
                risks.append((0.5 * z.max() + 0.5 * z.mean()).item())
        ordered = sorted(risks)
        self.assertGreater(ordered[1], ordered[0])
        self.assertGreater(ordered[3], ordered[2])
        rule = p.PrecisionPolicy(
            "model",
            4,
            config_fingerprint=p.transport_config_fingerprint(
                original.metadata.model_config_json
            ),
            tau_low=(ordered[0] + ordered[1]) / 2,
            tau_high=(ordered[2] + ordered[3]) / 2,
            v4_error_limit=10,
        )
        encoded = original.compress(rule)
        frozen = json.loads(encoded.metadata.transport_json)
        self.assertEqual(set(frozen["precision_tags"]), set(q.FORMATS))
        for actual, expected in zip(frozen["risks"], risks):
            self.assertAlmostEqual(actual, expected, places=12)
        original.release()
        env.pool.aliases.clear()
        encoded.restore(env.llm, list(range(13)), [], 0, 1)
        self.assertEqual(
            set(encoded.restore_statistics["format_bytes"]), set(q.FORMATS)
        )
        first_bytes = encoded.restore_statistics["transfer_bytes"]
        encoded.restore(env.llm, list(range(13)), [], 0, 1, repair_suffix_blocks=[0, 2])
        self.assertLess(encoded.restore_statistics["transfer_bytes"], first_bytes)
        self.assertEqual(encoded.restore_statistics["repaired_blocks"], [0, 2])
        self.assertEqual(json.loads(encoded.metadata.transport_json), frozen)
        encoded.release()

    def environment(self):
        import torch

        env = checkpoint_tests.CheckpointRecoveryTest().environment()
        env.session.core.vllm_config.model_config.dtype = torch.float16
        for layer in env.context.values():
            layer.kv_cache = layer.kv_cache.half()
        transport = transport_module()
        env.session.create.__func__.__globals__.update(
            create_quantized_checkpoint=transport.create_quantized_checkpoint,
            validate_transport_prefix=transport.validate_transport_prefix,
        )
        env.transport = transport
        return env

    def test_encoded_eviction_restore_mapping_rope_and_transfer_byte_accounting(self):
        import torch

        for fmt in q.FORMATS:
            env = self.environment()
            policy = p.PrecisionPolicy.experimental_fixed("model", 4, fmt)
            handle = env.session.create(
                list(range(13)),
                cache_salt="source-salt",
                model_key="model",
                precision_policy=policy,
                protected_prefix_tokens=3,
            )
            transport = json.loads(handle.metadata.transport_json)
            self.assertEqual(transport["precision_tags"], ["FP16", fmt, fmt, fmt])
            self.assertFalse(transport["quality_calibrated"])
            metadata_before = handle.metadata
            env.pool.aliases.clear()
            for layer in env.context.values():
                layer.kv_cache.fill_(float("nan"))
            lookups = env.pool.lookups
            handle.restore(
                env.llm, list(range(13)), [90], 3, 5, repair_suffix_blocks=[2]
            )
            self.assertEqual(env.pool.lookups, lookups)
            plan = planner.plan_suffix_repair(13, 3, 5, 1, 4, [2])
            mappings = c.checkpoint_copy_ranges(13, 3, 5, 1, plan)
            expected_bytes = 0
            for mapping in mappings:
                for index in range(
                    mapping.source_start, mapping.source_start + mapping.count
                ):
                    tag = transport["precision_tags"][index // 4]
                    expected_bytes += 2 * (32 if tag in ("FP16", "K8/V8") else 28)
            self.assertEqual(
                handle.restore_statistics["transfer_bytes"], expected_bytes
            )
            with env.session.store.lease(handle.checkpoint_id) as reader:
                for name, layer in env.context.items():
                    self.assertEqual(layer.kv_cache.dtype, torch.float16)
                    actual = torch.cat(
                        [
                            layer.kv_cache[page.block_id].transpose(0, 1)
                            for page in env.pool.allocated
                        ]
                    )[: plan.new_length]
                    for mapping in mappings:
                        expected = reader.read(
                            name, mapping.source_start, mapping.count
                        )
                        if mapping.source_start != mapping.destination_start:
                            env.session.restore.__func__.__globals__["_relocate_keys"](
                                expected[..., :4],
                                env.ropes[name],
                                torch.arange(
                                    mapping.source_start,
                                    mapping.source_start + mapping.count,
                                ),
                                torch.arange(
                                    mapping.destination_start,
                                    mapping.destination_start + mapping.count,
                                ),
                            )
                        torch.testing.assert_close(
                            actual[
                                mapping.destination_start : mapping.destination_start
                                + mapping.count
                            ],
                            expected,
                            atol=0,
                            rtol=0,
                        )
            self.assertEqual(handle.metadata, metadata_before)
            handle.release()
            self.assertEqual(env.session.store.used_bytes, 0)

    def test_compression_from_cpu_survives_original_release_and_prefix_guard(self):
        env = self.environment()
        original = env.session.create(
            list(range(13)), cache_salt="source-salt", model_key="model"
        )
        env.pool.aliases.clear()
        compressed = original.compress(
            p.PrecisionPolicy.experimental_fixed("model", 4, "K8/V4"),
            protected_prefix_tokens=3,
        )
        original.release()
        compressed.restore(env.llm, list(range(13)), [], 3, 5)
        with self.assertRaisesRegex(ValueError, "prefix"):
            compressed.restore(env.llm, list(range(13)), [], 6, 7)
        with self.assertRaises(ValueError):
            compressed.compress(
                p.PrecisionPolicy.experimental_fixed("model", 4, "FP16")
            )
        compressed.release()
        self.assertEqual(env.session.store.used_bytes, 0)

    def test_attention_risk_gate_and_metadata_roundtrip(self):
        env = self.environment()
        data = a.AttentionMetadata(
            "model",
            "source-salt",
            a.token_digest(range(13)),
            13,
            4,
            (12,),
            (a.AttentionLayerSpec("sample", 4, 2, 4, 0.5, "toy"),),
            tuple([1 / 13] * 13),
        )
        original = env.session.create(
            list(range(13)), cache_salt="source-salt", model_key="model", attention=data
        )
        candidate = p.PrecisionPolicy(
            "model",
            4,
            config_fingerprint=p.transport_config_fingerprint(
                original.metadata.model_config_json
            ),
            tau_low=100,
            tau_high=200,
            v4_error_limit=0,
        )
        compressed = original.compress(candidate, protected_prefix_tokens=3)
        saved = json.loads(compressed.metadata.transport_json)
        self.assertEqual(saved["precision_tags"], ["FP16", "K8/V8", "K8/V8", "K8/V8"])
        self.assertTrue(all(risk >= 0 for risk in saved["risks"][1:]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            compressed.metadata.save(path)
            self.assertEqual(c.CheckpointMetadata.load(path), compressed.metadata)
        profile, _, _ = checkpoint_tests.profile_tests.DriftPredictorTest().fitted()
        profile = replace(
            profile,
            config=replace(profile.config, model_key="model"),
            coefficients=(1.0, 0, 0, 0, 0, 0, 0),
            residual_margin=0,
            length_bounds=((0, 13), (0, 4), (0, 13)),
        )
        compressed.restore(
            env.llm,
            list(range(13)),
            [90],
            3,
            5,
            drift_profile=profile,
            attention_budget=a.AttentionBudget(max_tokens=1),
        )
        self.assertEqual(json.loads(compressed.metadata.transport_json), saved)
        original.release()
        compressed.release()

    def test_failed_encoding_reservation_and_incompatible_dtype_cleanup(self):

        env = self.environment()
        original = env.session.create(
            list(range(13)), cache_salt="source-salt", model_key="model"
        )
        used = env.session.store.used_bytes
        env.session.store.max_bytes = used + 1
        with self.assertRaises(MemoryError):
            original.compress(p.PrecisionPolicy.experimental_fixed("model", 4, "K8/V4"))
        self.assertEqual(env.session.store.used_bytes, used)
        env.session.store.max_bytes = 1 << 20
        metadata = replace(original.metadata, checkpoint_id="bad-copy")
        bad = Mock(side_effect=RuntimeError("encoding source failed"))
        with self.assertRaises(RuntimeError):
            env.transport.create_quantized_checkpoint(
                env.session.store,
                metadata,
                bad,
                p.PrecisionPolicy.experimental_fixed("model", 4, "K8/V8"),
                0,
            )
        self.assertEqual(env.session.store.used_bytes, used)
        encoded = original.compress(
            p.PrecisionPolicy.experimental_fixed("model", 4, "K8/V4"),
            protected_prefix_tokens=3,
        )
        env.pool.aliases.clear()
        env.fail_forward = True
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            encoded.restore(env.llm, list(range(13)), [90], 3, 5)
        self.assertFalse(env.pool.aliases)
        self.assertTrue(all(page.ref_cnt == 0 for page in env.pool.allocated))
        self.assertTrue(encoded.release())
        env.fail_forward = False
        original.release()
        self.assertEqual(env.session.store.used_bytes, 0)
        env = self.environment()
        for layer in env.context.values():
            layer.kv_cache = layer.kv_cache.float()
        with self.assertRaisesRegex(ValueError, "FP16"):
            env.session.create(
                list(range(13)),
                cache_salt="source-salt",
                model_key="model",
                precision_policy=p.PrecisionPolicy.experimental_fixed(
                    "model", 4, "FP16"
                ),
            )
        self.assertTrue(all(page.ref_cnt == 0 for page in env.pool.source))
        self.assertEqual(env.session.store.used_bytes, 0)

    def test_llm_checkpoint_api_forwards_codec_and_prefix(self):
        env = self.environment()
        env.llm._patchkv_checkpoint_session = env.session
        method = load_method("vllm/entrypoints/llm.py", "LLM", "create_kv_checkpoint")
        handle = method(
            env.llm,
            list(range(13)),
            model_key="model",
            cache_salt="source-salt",
            precision_policy=p.PrecisionPolicy.experimental_fixed("model", 4, "K8/V4"),
            protected_prefix_tokens=3,
        )
        self.assertEqual(
            json.loads(handle.metadata.transport_json)["precision_tags"][0], "FP16"
        )
        handle.release()


if __name__ == "__main__":
    unittest.main()
