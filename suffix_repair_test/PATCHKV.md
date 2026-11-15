# Manual PatchKV repair

This stage implements resident-KV manual repair only. There is no learned drift
policy, attention selection, CPU offload, or quantization.

## API

```python
edited = llm.edit_kv_cache(
    original_token_ids,
    "replacement text",
    start,
    end,
    cache_salt=original_salt,
    repair_suffix_blocks=[3, 5],
)
output = llm.generate(edited, sampling_params)
```

Block indices are **absolute logical block indices in the edited prompt**, not
physical page IDs or indices relative to the suffix. With block size 16, block 3
covers new positions `[48, 64)`. Every selected block must intersect the retained
suffix. Repair is clipped to that suffix; replacement tokens are always computed
and prefix tokens remain unchanged. Select valid indices for the actual prompt.

Adjacent selected blocks become one prefill interval. Noncontiguous intervals
execute in increasing position order, each with the full preceding destination
KV as its causal context. Reused shifted keys receive the existing RoPE
correction; values are copied unchanged. Shared source pages are never repair
destinations. Modified pages use copy-on-write and publish only after success.

- Omit `repair_suffix_blocks` to retain the existing direct-reuse behavior.
- Pass `repair_suffix_blocks=[]` for explicit manual mode with no suffix repair.
- Keep `recompute_suffix=True` for the existing full-suffix reference.
- Full recomputation and manual selection are mutually exclusive.

Selective repair is approximate: repaired tokens can attend to earlier reused
states. Full suffix recomputation, rather than selective repair, is the reference
for fresh-prefill numerical comparisons. Existing dense causal cache and 1D RoPE
restrictions still apply. Source KV must remain resident. The final prompt token
may run again during generation to produce logits, as in the existing editor.

## CPU checks

The planner uses only the standard library. These tests require no vLLM install:

```bash
.venv/bin/python -m unittest \
  suffix_repair_test.test_patchkv_planner \
  suffix_repair_test.test_patchkv_execution -v
```

On Windows, use `.venv/Scripts/python.exe`.

CPU tensor checks use PyTorch if it is already installed; otherwise they skip.
Do not install dependencies just to run these checks.

```bash
.venv/bin/python -m unittest discover -s suffix_repair_test \
  -p 'test_patchkv_*.py' -v
```

The mapping suite exhaustively checks 9,900 small edit/selection cases, including
insertions, deletions, replacements, truncations, empty results, and no-ops.
Control-flow tests load actual source definitions in isolation with fake cache
and model objects to avoid GPU-runtime imports. They cover page allocation,
cleanup, publication, dependency seeding, and API/RPC forwarding. Tensor tests
exercise the actual token-copy/RoPE functions and prefill metadata on CPU,
including both RoPE layouts, partial rotation, noncontiguous physical pages,
and chunking. They do not execute model attention or validate CUDA behavior.

Local validation on 2026-11-15 used the existing virtual environment:
all 22 PatchKV tests passed, including the 9,900 mapping cases and all block
subsets for representative insertion, deletion, and replacement edits.
Additional checks cover allocation failure cleanup and a CPU attention reference
with poisoned future rows across chunked and noncontiguous repair intervals.
Ruff lint/format checks, compilation of changed Python files, and
`git diff --check` passed. The original `test_kv_cache_edit` module could not
import in this minimal environment because vLLM runtime dependencies were not
installed (the first missing dependency was NumPy). Its independent RoPE/copy
numerical test passed through the isolated CPU tensor suite. No GPU was used.

## Later Colab validation

Use [the minimal Colab smoke guide](COLAB_PATCHKV.md) and
`patchkv_colab.py` for the first GPU session. The overlay helper transfers the
current uncommitted implementation without committing or pushing it.

Before benchmarking, use one small supported model and the same block size for
planning and execution. Compare full repair with a fresh-prefill reference using
appropriate floating-point tolerances. For sparse repair, record actual forward
positions and verify untouched source KV, reused values, relocated keys, and
decode completion. Include length-increasing/decreasing edits, separated repair
regions, a partial final block, chained edits, and both engine client modes.
Run the existing `suffix_repair_test.kv_cache_edit_gpu` regression alongside these
new cases. No GPU execution is part of the CPU checks above.
