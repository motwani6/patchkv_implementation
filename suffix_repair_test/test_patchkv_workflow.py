# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU orchestration checks; subprocess mocks never allocate a GPU."""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from suffix_repair_test import patchkv_workflow as workflow
from suffix_repair_test.patchkv_benchmark import summarize
from suffix_repair_test.patchkv_profile_tools import load_transport_support
from suffix_repair_test.test_patchkv_execution import load_definitions


class WorkflowTest(unittest.TestCase):
    def test_cpu_cli_preflight_and_planning_do_not_import_gpu_runtime(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            for module, options in (
                ("patchkv_environment", ["--output", str(root / "environment.json")]),
                ("patchkv_workflow", ["--results-dir", str(root / "plan")]),
            ):
                code = (
                    f"import sys; from suffix_repair_test.{module} import main; "
                    "main(); assert 'torch' not in sys.modules; "
                    "assert 'vllm' not in sys.modules"
                )
                result = subprocess.run(
                    [sys.executable, "-c", code, *options],
                    cwd=workflow.ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads((root / "environment.json").read_text())["status"],
                "cpu_audit_only",
            )
            plan = json.loads((root / "plan/workflow-plan.json").read_text())
            self.assertEqual(plan["status"], "planned_not_executed")
            self.assertFalse((root / "plan/workflow-results.json").exists())

    def fixture(self, root):
        def write(name, rows):
            (root / name).write_text(json.dumps(rows), encoding="utf-8")
            return name

        def task(name):
            return dict(
                task_id=name,
                original_ids=[1, 2, 3],
                start=1,
                end=2,
                replacement_text="new",
                expected_answer="answer",
            )

        return dict(
            drift=dict(
                pairs=write(
                    "pairs.json",
                    [
                        dict(
                            pair_id=name,
                            task_id=name,
                            original_ids=[1, 2, 3],
                            edited_ids=[1, 4, 3],
                        )
                        for name in ("drift-a", "drift-b")
                    ],
                ),
                tau=0.1,
                gamma=0.8,
                attention_fraction=0.1,
            ),
            precision=dict(
                tasks=write("quality.json", [task(n) for n in "abcd"]),
                candidates=write(
                    "grid.json",
                    [dict(alpha=0.5, tau_low=0.01, tau_high=0.1, v4_error_limit=0.02)],
                ),
                calibration_tasks=["a", "b"],
                validation_tasks=["c", "d"],
                max_quality_loss=0.01,
            ),
            benchmark=dict(tasks=write("held.json", [task("held")])),
        )

    def test_plan_order_and_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            plan = workflow.build_plan(self.fixture(root), root, root / "out")
            self.assertEqual(plan["status"], "planned_not_executed")
            self.assertTrue(plan["warnings"])
            self.assertEqual(plan["stages"][0]["name"], "00-environment")
            stage = plan["stages"][-1]
            self.assertEqual(stage["name"], "10-heldout-benchmark")
            self.assertEqual(
                stage["command"][stage["command"].index("--attention-fraction") + 1],
                "0.1",
            )
            self.assertEqual(len(plan["input_sha256"]), 4)

    def test_leakage_and_semantic_policy_rejected(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            manifest = self.fixture(root)
            manifest["benchmark"]["attention_fraction"] = 0.2
            with self.assertRaisesRegex(ValueError, "policy"):
                workflow.build_plan(manifest, root, root / "out")
            del manifest["benchmark"]["attention_fraction"]
            rows = json.loads((root / "held.json").read_text())
            rows[0]["task_id"] = "a"
            (root / "held.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "disjoint"):
                workflow.build_plan(manifest, root, root / "out")

    def test_invalid_inputs_rejected_before_execution(self):
        for change in (
            dict(start=True),
            dict(expected_answer=""),
            dict(original_ids=[-1]),
            dict(max_tokens=17),
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as value:
                root = Path(value)
                manifest = self.fixture(root)
                rows = json.loads((root / "held.json").read_text())
                rows[0].update(change)
                (root / "held.json").write_text(json.dumps(rows))
                with self.assertRaises(ValueError):
                    workflow.build_plan(manifest, root, root / "out")

    def test_fail_fast_saves_partial_results_and_logs(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as value:
                root = Path(value)
                plan = workflow.build_plan({}, root, root)
                calls = []

                def run(args, calls=calls, should_timeout=timeout, **options):
                    calls.append(args)
                    self.assertEqual(options["env"]["VLLM_USE_V2_MODEL_RUNNER"], "0")
                    self.assertLessEqual(options["timeout"], 5)
                    if should_timeout:
                        raise subprocess.TimeoutExpired(args, 5)
                    return SimpleNamespace(returncode=1)

                with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                    workflow.execute_plan(plan, root, stage_timeout=5, run=run)
                report = json.loads((root / "workflow-results.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertEqual(len(calls), 1)
                self.assertEqual(report["stages"][0]["status"], "failed")
                self.assertTrue((root / "logs/00-environment.log").exists())

    def test_changed_input_refused(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            plan = workflow.build_plan(self.fixture(root), root, root / "out")
            (root / "held.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "changed"):
                workflow.execute_plan(
                    plan, root / "out", run=lambda *a, **k: self.fail()
                )

    def test_benchmark_summary_pairs_by_task_and_repetition(self):
        # Explicit synthetic timing/quality fixtures, never experimental results.
        fields = dict(
            status="passed",
            task_id="fixture",
            repetition=0,
            recovery_ms=10,
            generation_ms=2,
            resume_to_response_ms=12,
            gpu_peak_allocated_bytes=100,
            cpu_payload_bytes=200,
            transfer_bytes=80,
            answer_token_f1=1,
            continuation_agreement=1,
        )
        rows = [
            dict(fields, mode="full"),
            dict(fields, mode="quantized", recovery_ms=5),
        ]
        result = summarize(rows)
        self.assertEqual(result["quantized"]["paired_recovery_speedup_median"], 2)
        self.assertEqual(result["full"]["observations"], 1)
        rows.append(dict(fields, mode="manual", status="failed"))
        self.assertNotIn("manual", summarize(rows))

    def test_transport_repair_scope_roundtrip_and_rejection(self):
        p = load_transport_support("patchkv_precision")
        scope = p.repair_policy_fingerprint({"profile": 1}, {"fraction": 0.1})
        policy = p.PrecisionPolicy(
            "model", 16, repair_fingerprint=scope, config_fingerprint="a" * 64
        )
        restored = p.PrecisionPolicy.from_dict(asdict(policy))
        self.assertEqual(restored.policy_id, policy.policy_id)
        namespace = dict(
            json=json,
            asdict=asdict,
            PrecisionPolicy=p.PrecisionPolicy,
            repair_policy_fingerprint=p.repair_policy_fingerprint,
        )
        load_definitions(
            "vllm/v1/core/patchkv_transport.py",
            ["validate_transport_repair"],
            namespace,
        )
        validate = namespace["validate_transport_repair"]
        metadata = SimpleNamespace(
            transport_json=json.dumps({"policy": asdict(policy)})
        )
        # Metadata contains canonical dataclass dictionaries, as at runtime.
        from dataclasses import make_dataclass

        Profile = make_dataclass("Profile", [("profile", int)])
        Budget = make_dataclass("Budget", [("fraction", float)])
        validate(metadata, Profile(1), Budget(0.1), "direct")
        with self.assertRaisesRegex(ValueError, "repair"):
            validate(metadata, Profile(1), Budget(0.2), "direct")
        with self.assertRaisesRegex(ValueError, "repair"):
            validate(metadata, None, None, "manual")
        self.assertNotEqual(
            policy.policy_id, replace(policy, repair_fingerprint=None).policy_id
        )


if __name__ == "__main__":
    unittest.main()
