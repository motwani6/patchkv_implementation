# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU control-flow tests using source definitions and fake cache/model objects.

These exercise the real editor and worker functions without importing the CUDA
runtime or vLLM dependencies. They do not validate GPU attention numerics.
"""

import ast
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from suffix_repair_test.test_patchkv_planner import planner

ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in definitions} == set(names)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module([future, *definitions], []))
    exec(compile(module, str(ROOT / path), "exec"), namespace)


def load_method(path, class_name, method_name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = {}
    module = ast.fix_missing_locations(ast.Module([future, method], []))
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[method_name]


class PatchKVAPIForwardingTest(unittest.TestCase):
    def test_llm_tokenizes_and_forwards_manual_blocks(self):
        method = load_method("vllm/entrypoints/llm.py", "LLM", "edit_kv_cache")
        core = SimpleNamespace(edit_kv_cache=Mock(return_value="edited-salt"))
        tokenizer = SimpleNamespace(encode=Mock(return_value=[90, 91]))
        llm = SimpleNamespace(
            get_tokenizer=lambda: tokenizer,
            llm_engine=SimpleNamespace(engine_core=core),
        )
        result = method(
            llm,
            list(range(26)),
            "new",
            5,
            7,
            cache_salt="old-salt",
            repair_suffix_blocks=(3, 5),
        )
        core.edit_kv_cache.assert_called_once_with(
            list(range(26)), [90, 91], 5, 7, "old-salt", False, [3, 5]
        )
        self.assertEqual(
            result["prompt_token_ids"], list(range(5)) + [90, 91] + list(range(7, 26))
        )
        self.assertEqual(result["cache_salt"], "edited-salt")

    def test_inproc_and_multiprocess_clients_forward_blocks(self):
        args = (list(range(26)), [90], 5, 7, None, False, [3, 5])
        for cls in ("InprocClient", "SyncMPClient"):
            with self.subTest(client=cls):
                method = load_method(
                    "vllm/v1/engine/core_client.py", cls, "edit_kv_cache"
                )
                call = Mock(return_value="salt")
                client = SimpleNamespace(
                    engine_core=SimpleNamespace(edit_kv_cache=call), call_utility=call
                )
                self.assertEqual(method(client, *args), "salt")
                if cls == "InprocClient":
                    call.assert_called_once_with(*args)
                else:
                    call.assert_called_once_with("edit_kv_cache", *args)

    def test_engine_core_forwards_planner_options(self):
        method = load_method("vllm/v1/engine/core.py", "EngineCore", "edit_kv_cache")
        editor = SimpleNamespace(edit=Mock(return_value="salt"))
        config = SimpleNamespace(max_model_len=128, get_vocab_size=lambda: 1000)
        core = SimpleNamespace(
            scheduler=SimpleNamespace(kv_cache_manager=SimpleNamespace(editor=editor)),
            vllm_config=SimpleNamespace(model_config=config),
            collective_rpc=Mock(),
        )
        method(core, list(range(26)), [90], 5, 7, None, repair_suffix_blocks=[3, 5])
        self.assertEqual(
            editor.edit.call_args.kwargs,
            {"recompute_suffix": False, "repair_suffix_blocks": [3, 5]},
        )
        editor.edit.call_args.args[-1]("plan")
        core.collective_rpc.assert_called_once_with(
            "write_kv_cache_edit", args=("plan",)
        )


@dataclass
class Page:
    block_id: int
    ref_cnt: int = 1


class FakePool:
    def __init__(self, count):
        self.source = [Page(i) for i in range(count)]
        self.allocated = []

    def get_cached_block(self, page, groups):
        return [self.source[page]]

    def touch(self, pages):
        for page in pages:
            page.ref_cnt += 1

    def get_new_blocks(self, count):
        self.allocated = [Page(100 + i) for i in range(count)]
        return self.allocated

    def free_blocks(self, pages):
        for page in pages:
            page.ref_cnt -= 1


EDITOR_NAMESPACE = {
    "dataclass": dataclass,
    "uuid4": uuid4,
    "plan_suffix_repair": planner.plan_suffix_repair,
    "cdiv": lambda n, d: (n + d - 1) // d,
}
load_definitions(
    "vllm/v1/core/kv_cache_edit.py",
    ["KVCacheEdit", "KVCacheEditor"],
    EDITOR_NAMESPACE,
)


class PatchKVEditorTest(unittest.TestCase):
    def make_editor(self, length=26):
        editor = EDITOR_NAMESPACE["KVCacheEditor"].__new__(
            EDITOR_NAMESPACE["KVCacheEditor"]
        )
        editor.block_size = 4
        editor.pool = FakePool((length + 3) // 4)
        editor._request = lambda ids, salt: SimpleNamespace(
            num_tokens=len(ids), ids=ids, salt=salt
        )
        editor._hash_at = lambda request, end: (end - 1) // 4
        editor._publish = Mock()
        return editor

    def test_noncontiguous_private_pages_and_publication(self):
        editor = self.make_editor()
        write = Mock()
        salt = editor.edit(
            list(range(26)),
            [90, 91],
            5,
            7,
            None,
            write,
            repair_suffix_blocks=[3, 5],
        )
        edit = write.call_args.args[0]
        self.assertEqual(edit.destination, [0, 100, 2, 101, 4, 102, 6])
        self.assertEqual(edit.recovery_plan.repair_ranges, ((12, 16), (20, 24)))
        self.assertEqual(editor._publish.call_count, 8)
        self.assertTrue(salt.startswith("kv-edit-"))
        self.assertTrue(all(p.ref_cnt == 1 for p in editor.pool.source))
        self.assertTrue(all(p.ref_cnt == 0 for p in editor.pool.allocated))

    def test_shifted_edits_privatize_suffix(self):
        for end, replacement in ((5, [90, 91]), (9, []), (7, [90])):
            with self.subTest(end=end, replacement=replacement):
                editor = self.make_editor()
                write = Mock()
                editor.edit(
                    list(range(26)),
                    replacement,
                    5,
                    end,
                    None,
                    write,
                    repair_suffix_blocks=[2, 4],
                )
                edit = write.call_args.args[0]
                self.assertEqual(edit.destination[0], 0)
                self.assertTrue(set(edit.source).isdisjoint(edit.destination[1:]))

    def test_legacy_modes_unchanged(self):
        for full in (False, True):
            with self.subTest(full=full):
                editor = self.make_editor()
                write = Mock()
                editor.edit(
                    list(range(26)),
                    [90, 91],
                    5,
                    7,
                    None,
                    write,
                    recompute_suffix=full,
                )
                edit = write.call_args.args[0]
                self.assertIsNone(edit.recovery_plan)
                self.assertEqual(edit.suffix_ids, list(range(7, 26)) if full else None)
                self.assertEqual(len(editor.pool.allocated), 6 if full else 1)

    def test_noop_and_empty_result_skip_worker(self):
        for start, end in ((5, 5), (0, 26), (6, 26)):
            with self.subTest(start=start, end=end):
                editor = self.make_editor()
                write = Mock()
                editor.edit(
                    list(range(26)),
                    [],
                    start,
                    end,
                    None,
                    write,
                    repair_suffix_blocks=[],
                )
                write.assert_not_called()

    def test_worker_failure_releases_pages_without_publication(self):
        editor = self.make_editor()
        with self.assertRaisesRegex(RuntimeError, "failed"):
            editor.edit(
                list(range(26)),
                [90],
                5,
                7,
                None,
                Mock(side_effect=RuntimeError("failed")),
                repair_suffix_blocks=[3, 5],
            )
        editor._publish.assert_not_called()
        self.assertTrue(all(p.ref_cnt == 1 for p in editor.pool.source))
        self.assertTrue(all(p.ref_cnt == 0 for p in editor.pool.allocated))

    def test_conflicting_modes_and_invalid_blocks_fail_before_allocation(self):
        for options in (
            dict(recompute_suffix=True, repair_suffix_blocks=[]),
            dict(repair_suffix_blocks=[0]),
        ):
            editor = self.make_editor()
            with self.assertRaises(ValueError):
                editor.edit(list(range(26)), [], 5, 7, None, Mock(), **options)
            self.assertEqual(editor.pool.allocated, [])

    def test_allocation_failure_releases_source_pins(self):
        editor = self.make_editor()
        editor.pool.get_new_blocks = Mock(side_effect=ValueError("out of pages"))
        write = Mock()
        with self.assertRaisesRegex(ValueError, "out of pages"):
            editor.edit(
                list(range(26)),
                [90],
                5,
                7,
                None,
                write,
                repair_suffix_blocks=[2, 4],
            )
        write.assert_not_called()
        editor._publish.assert_not_called()
        self.assertTrue(all(p.ref_cnt == 1 for p in editor.pool.source))


class FakeRope:
    head_size = 2
    rotary_dim = 2
    cos_sin_cache = SimpleNamespace(ndim=2, shape=(64, 2))

    def forward_native(self):
        pass


class PatchKVWorkerTest(unittest.TestCase):
    def test_selective_prefill_order_and_dependencies(self):
        for start, end, replacement, selected in (
            (5, 7, [90], [2, 4]),
            (5, 5, [90, 91], [2, 4]),
            (5, 9, [], [1, 4]),
            (5, 7, [90, 91], [1, 3, 5]),
            (5, 5, [], [1, 4]),
            (5, 7, [90], []),
        ):
            with self.subTest(start=start, end=end, replacement=replacement):
                self.check_execution(start, end, replacement, selected)

    def check_execution(self, start, end, replacement, selected):
        original = list(range(26))
        plan = planner.plan_suffix_repair(26, start, end, len(replacement), 4, selected)
        edited = original[:start] + replacement + original[end:]
        # Shared pages begin initialized. Private pages must be seeded or prefetched.
        initialized = {
            p for p in range(len(edited)) if p // 4 not in plan.private_blocks
        }
        events = []

        def copy(cache, rope, size, src, dst, old, new, count):
            events.append(("copy", new, new + count))
            self.assertEqual(original[old : old + count], edited[new : new + count])
            initialized.update(range(new, new + count))

        def prefill(runner, ids, pos, pages, size):
            if not ids:
                return
            events.append(("prefill", pos, pos + len(ids)))
            self.assertTrue(set(range(pos)) <= initialized)
            self.assertEqual(ids, edited[pos : pos + len(ids)])
            initialized.update(range(pos, pos + len(ids)))

        cache = SimpleNamespace(ndim=4, shape=(128, 1, 4, 4), dtype="fp16")
        cache.transpose = lambda a, b: cache
        rope = FakeRope()
        model = SimpleNamespace(
            get_submodule=lambda name: SimpleNamespace(rotary_emb=rope)
        )
        context = {"model.attn": SimpleNamespace(kv_cache=cache, num_kv_heads=1)}
        runner = SimpleNamespace(
            get_model=lambda: model,
            device="cpu",
            vllm_config=SimpleNamespace(
                compilation_config=SimpleNamespace(static_forward_context=context)
            ),
            attn_groups=[[SimpleNamespace(layer_names=["model.attn"])]],
        )
        namespace = {
            "torch": SimpleNamespace(
                inference_mode=lambda: lambda f: f,
                tensor=lambda x, **kw: x,
                long="long",
                float16="fp16",
                bfloat16="bf16",
                float32="fp32",
            ),
            "RotaryEmbedding": FakeRope,
            "_copy_tokens": copy,
            "_prefill": prefill,
        }
        load_definitions(
            "vllm/v1/worker/kv_cache_edit.py", ["write_kv_cache_edit"], namespace
        )
        edit = EDITOR_NAMESPACE["KVCacheEdit"](
            block_size=4,
            source=list(range(7)),
            destination=list(range(7)),
            copy_end=len(edited),
            replacement_ids=replacement,
            suffix_ids=original[end:],
            start=start,
            end=end,
            recovery_plan=plan,
        )
        namespace["write_kv_cache_edit"](runner, edit)
        expected_ranges = [(start, start + len(replacement))] if replacement else []
        expected_ranges += list(plan.repair_ranges)
        self.assertEqual(
            [(lo, hi) for kind, lo, hi in events if kind == "prefill"], expected_ranges
        )
        self.assertEqual(initialized, set(range(len(edited))))


if __name__ == "__main__":
    unittest.main()
