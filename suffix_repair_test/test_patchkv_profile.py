# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic CPU checks, not model calibration or experimental measurements."""

import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from suffix_repair_test.patchkv_profile_tools import load_profile_module
from suffix_repair_test.test_patchkv_planner import planner

p = load_profile_module()


def config(**kwargs):
    return p.DriftCalibration(
        model_key="synthetic/model@revision:fp16",
        layers=("layer.0",),
        block_size=4,
        deviation_threshold=0.1,
        stability_coverage=1.0,
        **kwargs,
    )


def sample(name, suffix=32, dirty=8, task=None, cfg=None):
    cfg = cfg or config()
    return p.DriftObservation(
        name,
        task or name,
        10,
        2,
        suffix,
        tuple(
            1.0 if index * cfg.block_size < dirty else 0.0
            for index in range(math.ceil(suffix / cfg.block_size))
        ),
        cfg,
    )


class DriftOracleTest(unittest.TestCase):
    def test_longest_prefix_and_nonoverlapping_suffix_differencing(self):
        for old, new, expected in (
            ((1, 2, 3), (1, 9, 2, 3), (1, 1, (9,))),
            ((1, 9, 2, 3), (1, 2, 3), (1, 2, ())),
            ((1, 2, 3), (1, 9, 3), (1, 2, (9,))),
            ((1, 1, 1), (1, 1), (2, 3, ())),
            ((1, 2), (1, 2), (2, 2, ())),
            ((), (1,), (0, 0, (1,))),
        ):
            pair = p.ProfilePair.from_contexts("pair", "task", old, new)
            self.assertEqual(pair.edited_ids, new)
            self.assertEqual((pair.start, pair.end, pair.replacement_ids), expected)

    def test_nonempty_tails_and_partial_block(self):
        cfg = config()
        for values, length, expected in (
            ((), 0, 0),
            ((0, 0), 7, 0),
            ((1, 0), 7, 4),
            ((0, 1), 7, 7),
            ((1, 1), 7, 7),
            ((1, 0, 1, 0), 15, 12),
        ):
            with self.subTest(values=values):
                self.assertEqual(p.oracle_dirty_window(values, length, cfg), expected)
        cfg = replace(cfg, stability_coverage=0.5)
        self.assertEqual(p.oracle_dirty_window((1, 0, 1, 0), 15, cfg), 0)

    def test_invalid_calibration_and_deviations(self):
        for change in (
            dict(block_size=0),
            dict(block_size=True),
            dict(layers=()),
            dict(layers=("x", "x")),
            dict(deviation_threshold=float("nan")),
            dict(stability_coverage=0),
            dict(aggregate_quantile=1.1),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(config(), **change)
        for values, length in (((0,), 0), ((float("nan"),), 1), ((-1,), 1)):
            with self.assertRaises(ValueError):
                p.oracle_dirty_window(values, length, config())

    def test_task_split_preserves_adjacent_revisions(self):
        samples = [
            sample(f"pair-{i}-{j}", task=f"task-{i}")
            for i in range(6)
            for j in range(2)
        ]
        training, calibration = p.split_by_task(samples, seed=42)
        self.assertFalse(
            {s.task_id for s in training} & {s.task_id for s in calibration}
        )
        self.assertEqual(len(training) + len(calibration), len(samples))
        self.assertEqual((training, calibration), p.split_by_task(samples, seed=42))
        with self.assertRaises(ValueError):
            p.split_by_task(samples[:2])


class DriftPredictorTest(unittest.TestCase):
    def fitted(self):
        training = [
            sample(f"train-{i}", suffix=length, dirty=length // 2)
            for i, length in enumerate(range(16, 81, 8))
        ]
        calibration = [
            sample(f"cal-{i}", suffix=length, dirty=length // 2 + 4)
            for i, length in enumerate(range(16, 81, 8))
        ]
        return (
            p.fit_drift_profile(training, calibration, quantile=0.8),
            training,
            calibration,
        )

    def test_fit_length_dependence_and_task_calibration(self):
        profile, training, calibration = self.fitted()
        self.assertIsNotNone(profile.residual_margin)
        self.assertLess(profile.predict(10, 2, 16), profile.predict(10, 2, 80))
        covered = sum(
            profile.predict(*s.lengths) >= s.dirty_window for s in calibration
        )
        self.assertGreaterEqual(covered / len(calibration), 0.8)
        self.assertTrue(all(math.isfinite(c) for c in profile.coefficients))
        # A trained length model must improve over the constant upper-quantile loss.
        targets = sorted(s.dirty_window for s in training)
        intercept = targets[math.ceil(0.8 * len(targets)) - 1]
        constant_loss = sum(
            max(0.8 * (y - intercept), -0.2 * (y - intercept)) for y in targets
        ) / len(targets)
        self.assertLess(profile.training_pinball_loss, constant_loss)

    def test_fallback_for_small_calibration_and_extrapolation(self):
        profile = p.fit_drift_profile([sample("train")], [sample("cal")])
        self.assertIsNone(profile.residual_margin)
        self.assertEqual(profile.predict(10, 2, 32), 32)
        profile, _, _ = self.fitted()
        for lengths in ((11, 2, 32), (10, 3, 32), (10, 2, 200), (0, 0, 0)):
            self.assertEqual(profile.predict(*lengths), lengths[-1])
        with self.assertRaises(ValueError):
            profile.predict(1, 1, -1)

    def test_group_max_residuals_not_pair_count(self):
        calibration = [sample(f"cal-{i}", task="same-task") for i in range(30)]
        profile = p.fit_drift_profile([sample("train")], calibration, quantile=0.8)
        self.assertIsNone(profile.residual_margin)

    def test_reject_data_leakage_and_mixed_configs(self):
        for training, calibration in (
            ([sample("same")], [sample("same")]),
            ([sample("a", task="same")], [sample("b", task="same")]),
            ([sample("a")], [sample("b", cfg=replace(config(), model_key="other"))]),
            ([], [sample("b")]),
        ):
            with self.assertRaises(ValueError):
                p.fit_drift_profile(training, calibration)

    def test_profile_and_observations_roundtrip_and_schema_validation(self):
        profile, training, calibration = self.fitted()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile.save(path)
            self.assertEqual(profile, p.DriftProfile.load(path))
            p.save_observations(path, [*training, *calibration])
            self.assertEqual([*training, *calibration], p.load_observations(path))
            profile.save(path)
            data = json.loads(path.read_text())
            for field, value in (
                ("schema_version", 999),
                ("coefficients", [float("nan")] * 7),
                ("feature_scales", [0] * 7),
                ("residual_margin", -1),
            ):
                path.write_text(json.dumps({**data, field: value}))
                with self.assertRaises(ValueError):
                    p.DriftProfile.load(path)

    def test_planner_rounds_once_and_unions_nonlocal_blocks(self):
        profile, _, _ = self.fitted()
        for original_length, start, end, replacement in (
            (44, 10, 12, 2),
            (42, 10, 10, 2),
            (46, 10, 14, 0),
        ):
            plan = planner.plan_profiled_suffix_repair(
                original_length,
                start,
                end,
                replacement,
                4,
                profile,
                model_key=profile.config.model_key,
                additional_repair_suffix_blocks=[
                    (start + replacement + original_length - end - 1) // 4
                ],
            )
            dirty = profile.predict(start, replacement, original_length - end)
            expected = {
                i // 4 for i in range(start + replacement, start + replacement + dirty)
            }
            expected.add((plan.new_length - 1) // 4)
            self.assertEqual(set(plan.repair_blocks), expected)
        for size, key in ((8, profile.config.model_key), (4, "wrong-model")):
            with self.assertRaises(ValueError):
                planner.plan_profiled_suffix_repair(
                    44, 10, 12, 2, size, profile, model_key=key
                )

    def test_cpu_fit_cli_and_independent_validation(self):
        from suffix_repair_test.patchkv_profile_cli import fit

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations.json"
            validation = root / "validation.json"
            p.save_observations(observations, [sample(f"fit-{i}") for i in range(8)])
            p.save_observations(validation, [sample("held-out")])
            args = Namespace(
                observations=observations,
                validation=validation,
                output_dir=root / "fitted",
                quantile=0.8,
                calibration_fraction=0.5,
                seed=0,
            )
            with redirect_stdout(io.StringIO()):
                fit(args)
            profile = p.DriftProfile.load(args.output_dir / "profile.json")
            self.assertIsNotNone(profile.residual_margin)
            report = json.loads((args.output_dir / "fit_summary.json").read_text())
            self.assertEqual(report["validation"]["boundary_coverage"], 1)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "from suffix_repair_test.patchkv_profile_cli import main; "
                        "main(); assert 'torch' not in sys.modules; "
                        "assert 'vllm' not in sys.modules"
                    ),
                    "fit",
                    "--observations",
                    str(observations),
                    "--validation",
                    str(validation),
                    "--output-dir",
                    str(root / "subprocess-fit"),
                    "--quantile",
                    "0.8",
                    "--calibration-fraction",
                    "0.5",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            p.save_observations(validation, [sample("fit-0")])
            with self.assertRaises(ValueError):
                fit(replace_namespace(args, output_dir=root / "leaked"))


def replace_namespace(namespace, **changes):
    return Namespace(**{**vars(namespace), **changes})


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch unavailable")
class DriftTensorTest(unittest.TestCase):
    def test_cosines_alignment_both_shift_signs_and_partial_blocks(self):
        import torch

        for start, end, replacement in ((2, 2, (90, 91)), (2, 5, ()), (2, 4, (90,))):
            pair = p.ProfilePair(
                "pair", "task", tuple(range(12)), start, end, replacement
            )
            old = torch.zeros(12, 1, 2)
            old[:, 0, 0] = 1
            new = torch.zeros(len(pair.edited_ids), 1, 2)
            new[:, 0, 0] = 1
            suffix_start = start + len(replacement)
            new[suffix_start : suffix_start + 4] *= -1
            observed = p.observe_values(
                pair, {"layer.0": old}, {"layer.0": new}, config()
            )
            self.assertEqual(observed.block_deviations[0], 2)
            self.assertTrue(all(x == 0 for x in observed.block_deviations[1:]))
            self.assertEqual(observed.dirty_window, 4)

    def test_zero_norms_scaling_and_robust_layer_pooling(self):
        import torch

        pair = p.ProfilePair("pair", "task", (1, 2, 3, 4), 0, 0, ())
        old = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]])
        new = torch.tensor([[[0.0, 0.0]], [[1.0, 0.0]], [[9.0, 0.0]], [[0.0, 1.0]]])
        observed = p.observe_values(pair, {"layer.0": old}, {"layer.0": new}, config())
        self.assertEqual(observed.block_deviations, (1,))
        cfg = replace(config(), layers=("layer.0", "layer.1"), aggregate_quantile=0.5)
        observed = p.observe_values(
            pair,
            {"layer.0": old, "layer.1": old},
            {"layer.0": old, "layer.1": new},
            cfg,
        )
        self.assertEqual(observed.block_deviations, (0,))

    def test_prefill_adapter_called_twice_and_old_buffer_is_copied(self):
        import torch

        pair = p.ProfilePair("pair", "task", (1, 2, 3, 4), 0, 1, (90,))
        buffer = torch.ones(4, 1, 2)
        calls = []

        def prefill(ids):
            calls.append(ids)
            if len(calls) == 2:
                buffer.neg_()
            return {"layer.0": buffer}

        observed = p.profile_pair(pair, prefill, config())
        self.assertEqual(calls, [pair.original_ids, pair.edited_ids])
        self.assertAlmostEqual(observed.block_deviations[0], 2)
        self.assertEqual(observed.dirty_window, 3)

    def test_empty_suffix_and_invalid_tensor_inputs(self):
        import torch

        pair = p.ProfilePair("pair", "task", (1, 2), 0, 2, ())
        observed = p.observe_values(
            pair,
            {"layer.0": torch.zeros(2, 1, 2)},
            {"layer.0": torch.zeros(0, 1, 2)},
            config(),
        )
        self.assertEqual(observed.dirty_window, 0)
        pair = p.ProfilePair("pair", "task", (1, 2), 0, 0, ())
        valid = torch.ones(2, 1, 2)
        for invalid in (
            torch.ones(1, 1, 2),
            torch.ones(2, 2),
            torch.full((2, 1, 2), float("nan")),
            valid.clone().requires_grad_(),
            valid.long(),
        ):
            with self.assertRaises(ValueError):
                p.observe_values(
                    pair, {"layer.0": valid}, {"layer.0": invalid}, config()
                )
        with self.assertRaises(ValueError):
            p.observe_values(pair, {}, {"layer.0": valid}, config())


if __name__ == "__main__":
    unittest.main()
