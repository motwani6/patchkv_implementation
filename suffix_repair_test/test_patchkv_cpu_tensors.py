# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tensor checks; requires CPU PyTorch but no installed vLLM or GPU."""

import importlib.util
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

from suffix_repair_test.test_patchkv_execution import load_definitions


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch is optional")
class PatchKVCPUTensorTest(unittest.TestCase):
    def test_existing_rope_and_page_copy_numerics(self):
        import torch

        class CustomOp:
            @classmethod
            def register(cls, name):
                return lambda implementation: implementation

        namespace = {"torch": torch, "CustomOp": CustomOp}
        load_definitions(
            "vllm/model_executor/layers/rotary_embedding/common.py",
            ["ApplyRotaryEmb"],
            namespace,
        )
        load_definitions(
            "vllm/v1/worker/kv_cache_edit.py",
            ["_relocate_keys", "_copy_tokens"],
            namespace,
        )
        namespace.update(unittest=unittest, SimpleNamespace=SimpleNamespace)
        load_definitions(
            "suffix_repair_test/test_kv_cache_edit.py", ["KVCacheEditTest"], namespace
        )
        # Reuse the existing independent RoPE reference, without CUDA imports.
        case = namespace["KVCacheEditTest"]
        case.test_rope_layouts_partial_rotation_and_noncontiguous_pages(self)

    def test_prefill_uses_absolute_positions_and_causal_lengths(self):
        import torch

        records = []
        metadata = []

        def build_metadata(*args, **kwargs):
            metadata.append(kwargs)
            return kwargs

        def forward(input_ids, positions):
            records.append((input_ids.tolist(), positions.tolist()))

        runner = SimpleNamespace(
            device="cpu",
            get_model=lambda: forward,
            kv_cache_config=object(),
            attn_groups=object(),
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_batched_tokens=3)
            ),
        )
        namespace = {
            "torch": torch,
            "cdiv": lambda n, d: (n + d - 1) // d,
            "prepare_kernel_block_sizes": lambda *args: (2,),
            "build_attn_metadata": build_metadata,
            "set_forward_context": lambda *args, **kwargs: nullcontext(),
            "build_slot_mappings_by_layer": lambda *args: {},
        }
        load_definitions("vllm/v1/worker/kv_cache_edit.py", ["_prefill"], namespace)
        pages = torch.tensor([8, 3, 9, 2, 7, 5])
        namespace["_prefill"](runner, [90, 91, 92, 93], 12, pages, 4)
        namespace["_prefill"](runner, [94, 95, 96, 97], 20, pages, 4)
        self.assertEqual(
            records,
            [
                ([90, 91, 92], [12, 13, 14]),
                ([93], [15]),
                ([94, 95, 96], [20, 21, 22]),
                ([97], [23]),
            ],
        )
        self.assertEqual([m["max_seq_len"] for m in metadata], [15, 16, 23, 24])
        for record, item in zip(records, metadata):
            self.assertIs(item["causal"], True)
            positions = torch.tensor(record[1])
            expected = pages[positions // 4] * 4 + positions % 4
            torch.testing.assert_close(item["slot_mappings"][0], expected)
            self.assertEqual(item["seq_lens"].item(), record[1][-1] + 1)

    def test_prefill_causal_attention_excludes_future_reused_states(self):
        import torch

        # Deliberately poison future rows: a missing causal mask changes outputs.
        values = torch.full((32,), 10000.0)
        values[:5] = torch.arange(5, dtype=torch.float32)
        active = {}

        def build_metadata(*args, **kwargs):
            active.update(kwargs)
            return kwargs

        def forward(input_ids, positions):
            count = len(input_ids)
            end = active["seq_lens"].item()
            values[positions] = input_ids.float()
            scores = torch.zeros(count, end)
            if active["causal"]:
                mask = torch.arange(end)[None, :] > positions[:, None]
                scores.masked_fill_(mask, -torch.inf)
            outputs.extend((scores.softmax(-1) @ values[:end]).tolist())

        runner = SimpleNamespace(
            device="cpu",
            get_model=lambda: forward,
            kv_cache_config=object(),
            attn_groups=object(),
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_batched_tokens=3)
            ),
        )
        namespace = {
            "torch": torch,
            "cdiv": lambda n, d: (n + d - 1) // d,
            "prepare_kernel_block_sizes": lambda *args: (4,),
            "build_attn_metadata": build_metadata,
            "set_forward_context": lambda *args, **kwargs: nullcontext(),
            "build_slot_mappings_by_layer": lambda *args: {},
        }
        load_definitions("vllm/v1/worker/kv_cache_edit.py", ["_prefill"], namespace)
        outputs = []
        pages = torch.arange(8)
        namespace["_prefill"](runner, [10, 11, 12, 13], 5, pages, 4)
        # A noncontiguous repair must include the intervening reused states.
        values[9:12] = torch.tensor([20.0, 21.0, 22.0])
        namespace["_prefill"](runner, [30, 31], 12, pages, 4)
        expected = [values[:end].mean() for end in (6, 7, 8, 9, 13, 14)]
        torch.testing.assert_close(torch.tensor(outputs), torch.stack(expected))


if __name__ == "__main__":
    unittest.main()
