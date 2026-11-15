# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with .venv/bin/python -m unittest suffix_repair_test.test_patchkv_planner."""

import importlib.util
import itertools
import sys
import unittest
from pathlib import Path

# Load the pure planner without importing vLLM or its optional dependencies.
PLANNER_PATH = Path(__file__).resolve().parents[1] / "vllm/v1/core/patchkv.py"
SPEC = importlib.util.spec_from_file_location(
    "patchkv_planner_under_test", PLANNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)
plan_suffix_repair = planner.plan_suffix_repair


class PatchKVPlannerTest(unittest.TestCase):
    def test_noncontiguous_and_adjacent_blocks(self):
        plan = plan_suffix_repair(25, 3, 6, 5, 4, [6, 2, 3, 2])
        self.assertEqual(plan.suffix_start, 8)
        self.assertEqual(plan.new_length, 27)
        self.assertEqual(plan.repair_blocks, (2, 3, 6))
        self.assertEqual(plan.reuse_blocks, (4, 5))
        self.assertEqual(plan.repair_ranges, ((8, 16), (24, 27)))
        self.assertEqual(plan.private_blocks, tuple(range(7)))

    def test_unshifted_repair_only_privatizes_selected_pages(self):
        plan = plan_suffix_repair(26, 5, 7, 2, 4, [3, 5])
        self.assertEqual(plan.private_blocks, (1, 3, 5))
        self.assertEqual(plan.repair_ranges, ((12, 16), (20, 24)))
        self.assertEqual(
            [(c.source_start, c.destination_start, c.count) for c in plan.copies],
            [(4, 4, 1), (7, 7, 1)],
        )

    def test_boundary_block_clips_repair_to_suffix(self):
        plan = plan_suffix_repair(14, 5, 7, 1, 4, [1, 3])
        self.assertEqual(plan.repair_ranges, ((6, 8), (12, 13)))
        self.assertEqual(plan.copies[0], planner.KVTokenCopy(4, 4, 1))

    def test_noop_and_truncation_need_no_private_pages(self):
        for args in ((14, 5, 5, 0), (14, 6, 14, 0), (14, 0, 14, 0)):
            with self.subTest(args=args):
                plan = plan_suffix_repair(*args, 4, [])
                self.assertEqual(plan.private_blocks, ())
                self.assertEqual(plan.copies, ())

    def test_noop_can_repair_noncontiguous_suffix(self):
        plan = plan_suffix_repair(20, 5, 5, 0, 4, [1, 4])
        self.assertEqual(plan.private_blocks, (1, 4))
        self.assertEqual(plan.repair_ranges, ((5, 8), (16, 20)))

    def test_invalid_inputs(self):
        for args in (
            (-1, 0, 0, 0, 4, []),
            (10, 4, 3, 0, 4, []),
            (10, 0, 11, 0, 4, []),
            (10, 0, 1, -1, 4, []),
            (10, 0, 1, 0, 0, []),
            (10, 4, 8, 1, 4, [0]),
            (10, 4, 8, 1, 4, [-1]),
            (10, 4, 8, 1, 4, [2]),
            (10, 4, 8, 1, 4, [True]),
            (10, 4, 8, 1, 4, [1.0]),
            (10, 4, 10, 1, 4, [1]),
            (10.0, 0, 1, 0, 4, []),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                plan_suffix_repair(*args)

    def test_exhaustive_token_coverage_and_mapping(self):
        cases = 0
        for length in range(9):
            for start in range(length + 1):
                for end in range(start, length + 1):
                    for replacement in range(5):
                        for size in (1, 2, 3, 4):
                            suffix_start = start + replacement
                            new_length = suffix_start + length - end
                            blocks = sorted(
                                {p // size for p in range(suffix_start, new_length)}
                            )
                            selections = ([], blocks, blocks[::2])
                            for selected in selections:
                                plan = plan_suffix_repair(
                                    length, start, end, replacement, size, selected
                                )
                                self.check_mapping(
                                    plan, length, start, end, replacement, selected
                                )
                                cases += 1
        self.assertEqual(cases, 9900)

    def test_every_subset_of_suffix_blocks(self):
        for start, end, replacement in ((1, 1, 3), (1, 4, 0), (1, 3, 2)):
            suffix_start = start + replacement
            new_length = suffix_start + 12 - end
            blocks = sorted({p // 2 for p in range(suffix_start, new_length)})
            for flags in itertools.product((False, True), repeat=len(blocks)):
                selected = [b for b, flag in zip(blocks, flags) if flag]
                with self.subTest(edit=(start, end, replacement), selected=selected):
                    plan = plan_suffix_repair(12, start, end, replacement, 2, selected)
                    self.check_mapping(plan, 12, start, end, replacement, selected)

    def check_mapping(self, plan, length, start, end, replacement, selected):
        size = plan.block_size
        original = list(range(length))
        expected = original[:start] + [None] * replacement + original[end:]
        repaired = set(
            itertools.chain.from_iterable(
                range(lo, hi) for lo, hi in plan.repair_ranges
            )
        )
        expected_repaired = {
            p for p in range(plan.suffix_start, len(expected)) if p // size in selected
        }
        self.assertEqual(repaired, expected_repaired)
        self.assertFalse(set(plan.repair_blocks) & set(plan.reuse_blocks))
        private = set(plan.private_blocks)
        writes = set(range(start, plan.suffix_start)) | repaired
        actual = {
            p: original[p] for p in range(len(expected)) if p // size not in private
        }
        for mapping in plan.copies:
            for offset in range(mapping.count):
                src = mapping.source_start + offset
                dst = mapping.destination_start + offset
                self.assertTrue(0 <= src < length)
                self.assertTrue(0 <= dst < len(expected))
                self.assertIn(dst // size, private)
                self.assertNotIn(dst, actual)
                self.assertNotIn(dst, writes)
                actual[dst] = original[src]
        for pos in writes:
            self.assertIn(pos // size, private)
            self.assertNotIn(pos, actual)
            actual[pos] = expected[pos]
        self.assertEqual([actual[p] for p in range(len(expected))], expected)


if __name__ == "__main__":
    unittest.main()
