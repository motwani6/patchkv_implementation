# Phase 4: bounded CPU KV checkpoints

Phase 4 implements in-memory CPU KV storage, creation/restoration/release, and
recovery after all source GPU pages are evicted. It retains source FP16/BF16/FP32
precision. Phase 5 now extends this baseline with optional encoded checkpoint
storage; see [the Phase 5 guide](PATCHKV_PHASE5.md). The baseline remains unchanged.
Phases 1–3 and all existing resident editing modes remain available.

This implements the CPU-resident checkpoint and old-to-new reconstruction parts
of [Section IV](https://arxiv.org/html/2609.26219v1#S4). Compressed representations,
clustered transport, and fused quantize/dequantize kernels remain separate work.

## API

Use an idle in-process LLM (`VLLM_ENABLE_V1_MULTIPROCESSING=0` before engine
creation), with all parallelism sizes equal to one. Callers must serialize
generation/checkpoint/edit calls: the checkpoint session lock does not coordinate
unrelated concurrent `generate` calls.

```python
checkpoint = llm.create_kv_checkpoint(
    original_ids, cache_salt=source_salt,
    model_key=model_revision_dtype_backend_key,
    max_cpu_bytes=64 << 20,
    attention=attention_metadata,  # Optional matching Phase 3 metadata.
)
try:
    # Normal cache pressure can evict the source, or explicitly:
    assert llm.reset_prefix_cache()
    edited = llm.edit_kv_cache(
        original_ids, replacement_text, start, end,
        checkpoint=checkpoint, repair_suffix_blocks=[3, 5],
    )
    output = llm.generate(edited, sampling_params)
finally:
    checkpoint.release()
```

Use valid absolute destination logical block indices for the actual prompt.
Omitting repair options computes replacement tokens and restores all suffix KV.
`recompute_suffix=True` recomputes the complete suffix. Alternatively, pass
`drift_profile=profile`, optionally with
`attention_budget=AttentionBudget(fraction=...)`. Attention-guided recovery
requires matching attention metadata supplied during checkpoint creation.
Full/manual/predicted modes are mutually exclusive. The attention metadata retains
its original source cache salt; the CPU snapshot receives a separate unique ID.

`checkpoint.checkpoint_id` identifies CPU ownership; `checkpoint.metadata` returns
immutable metadata. Release returns true once and false thereafter; subsequent
metadata reads/restores fail. Handles also support context-manager release.
Releasing a snapshot does not invalidate published destination GPU pages. A
restored prompt can be checkpointed using its returned salt for chained edits.
Handles are bound to the LLM instance that created them.

One per-LLM store shares a memory budget across all handles. The first call fixes
the budget (default 256 MiB). Later calls may omit `max_cpu_bytes` to reuse it;
an explicitly different budget fails. There is no silent CPU LRU eviction.
Release unneeded handles explicitly. Leases prevent release during a restore;
the same live handle can be restored repeatedly.

## Representation and compatibility

`vllm/v1/core/patchkv_checkpoint.py` owns one contiguous CPU tensor per layer,
shaped `[original_tokens, kv_heads, 2 * head_dim]`, containing stored rotated K
followed by V. Source dtype is retained exactly, without promotion or lossy
encoding. Partial final pages store valid rows only, never padding.

Metadata preserves original token IDs, unique checkpoint ID, source salt,
declared model key, actual model configuration/revision/dtype, block size, layer
names/head shapes/dtypes, and RoPE signatures. Every logical block records its
index, original absolute start, and valid row count. Positions are contiguous
from zero. **Neither CPU payload nor metadata stores physical GPU page IDs.**
Temporary physical page tables exist only while gathering or restoring.

Restore validates actual engine configuration, layer geometry/dtype, RoPE
signatures, original IDs, and block size. An explicitly supplied source salt must
also match. The model key must match supplied profiles. RoPE signatures include
implementation/layout and representative table rows; complete configuration and
revision metadata provide the remaining identity checks. In-place model weight
updates are unsupported and cannot be detected merely by configuration equality.
KV inherited from an earlier approximate edit remains approximate.

`checkpoint.metadata.save(path)` and `CheckpointMetadata.load(path)` round-trip
versioned data-only JSON metadata. **These methods do not save tensor payloads.**
This phase provides process-local CPU storage, not disk persistence, runtime
restart recovery, or engine migration. Metadata JSON alone cannot restore KV.

## Ownership and failure handling

Creation resolves resident source pages by salted hashes and pins them only
while copying. CPU memory is reserved before tensor allocation/KV gathering.
Transfers are blocking pageable D2H copies; device synchronization completes
before ownership is returned or source pins are released. Creation does not
itself clear the GPU cache, but all source pages become ordinary eviction
candidates afterward. Copy/budget/synchronization failures release the pins and
roll back CPU ownership/accounting.

Recovery performs no original-page lookup. It leases CPU storage and allocates
every destination page privately, including prefix pages that resident Phase 1
would share. This is checkpoint copy-on-write: another cached version is never a
write destination. Any physical IDs may be used, including recycled source IDs.

The restore map seeds the entire unchanged prefix and every reusable suffix
interval. All layers' dependencies are placed before any forward. Shifted keys
receive `R(new) * inverse(R(old))`; values are never rotated. Bounded CPU reads
return private clones, so relocation cannot alter stored KV. Replacement and
selected repair intervals then use the existing absolute-position causal prefill
path in increasing order. Generation may rerun the final prompt token for logits.

Device synchronization precedes reconstruction and completes before publication
or recycling. Allocation/transfer/prefill failures publish no aliases. If alias
publication fails partway, all aliases on the new private pages are removed
before those pages are released. Leases close and the CPU snapshot survives for
retry. Fatal CUDA errors that make synchronization unusable still require
discarding the engine/runtime; recovery from a poisoned CUDA context is not
claimed by the CPU tests.

The entrypoint directly uses an idle local engine/worker; multiprocess CPU-payload
RPC, pinned asynchronous offload, and overlapping transport are not implemented.
Supported layouts are the Phase 1 single-group dense causal unquantized cache
with supported native 1D RoPE and `TRITON_ATTN`. Distributed execution, KV transfer,
LoRA, speculative decoding, quantized GPU attention KV, hybrid/sliding-window caches, multimodal
prompts, shared-KV layers, arbitrary position IDs/MRoPE, and alternate layouts
are unsupported. Automatic backend selection is not assumed compatible.

## Memory cost and bounds

For N valid tokens, layer l with H_l KV heads, size D_l, and element bytes s_l:

`payload_bytes = N * sum_l(2 * H_l * D_l * s_l)`

For example, N=8192, 24 layers, 2 KV heads, head size 64, and FP16 require 96 MiB
of payload (calculated, not measured). `metadata.payload_bytes` reports tensor
bytes. `CPUCheckpointStore.charged_bytes(metadata)` adds recursive Python metadata
size and a conservative 4096-byte ownership allowance per layer plus 4096 bytes
per checkpoint. All entries and an in-progress reservation count against the
store budget; failed creation rolls accounting back. Capture never makes a
second full-context CPU KV copy.

Transfer scratch is bounded to 256 rows per layer by default, plus page-table
indices/framework overhead. Restore processes one layer/interval chunk at a time.
The budget excludes caller-owned inputs/exported copies, temporary metadata/JSON
construction, model prefill allocations, framework allocator caching, and runtime
overhead: it is a bounded store, not an OS RSS limit. Allow additional process
headroom for those costs. Recovery also needs GPU capacity for every destination
page, even when most rows are restored rather than recomputed.

## CPU validation and prepared Colab tests

```powershell
.venv/Scripts/python.exe -m unittest discover -s suffix_repair_test -p 'test_patchkv_*.py' -v
.venv/Scripts/python.exe -m suffix_repair_test.patchkv_checkpoint_colab --help
```

On 2026-11-15 all 65 CPU tests passed: 52 preserved tests plus 13 new checkpoint
tests. Coverage includes logical mapping, exhaustive small edits, dtype/ownership,
bounded reads, lifecycle/leases, budget rejection, copy/synchronization failure,
configuration/RoPE mismatch, allocation/prefill/publication rollback and retry,
complete source eviction/overwrite, causal repair, both shift signs, partial and
interleaved RoPE, direct/full/predicted/attention modes, chained snapshots, and API
forwarding. Synthetic eviction tests clear source aliases, overwrite physical
cache with NaNs, and assert restore makes no original-page lookup. These tests
exercise actual restore functions on CPU tensors, not CUDA.

After the `COLAB_PATCHKV.md` environment preflight, run one small GPU smoke suite:

```bash
%%bash
set -euo pipefail
cd /content/vllm-suffix-repair
RESULT=/content/drive/MyDrive/patchkv-checkpoints-$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python -m suffix_repair_test.patchkv_checkpoint_colab \
  --results-dir "$RESULT" \
  2>&1 | tee /content/drive/MyDrive/patchkv-checkpoints-run.log
```

It uses one pinned Qwen2.5-0.5B load, FP16/eager/legacy runner, Triton, 83-token
sources, 32-token prefill chunks, 32 MiB GPU KV and a 16 MiB CPU budget. Each
insertion/deletion/replacement case creates a snapshot, evicts all source aliases,
overwrites every original physical source page with NaNs (leaving the null page
untouched), restores direct/selective/full modes, and generates two tokens.
Checks include forward positions, exact prefix/value preservation, independent
RoPE relocation, full-repair versus fresh-prefill tolerances, cache reuse, and
accounting release. Progress is saved after each mode in `checkpoint-results.json`,
with metadata JSON, configuration, source hashes, and failure tracebacks.

Add `--drift-profile /path/to/profile.json` to test predicted restore and
`--attention-fraction 0.05` to also test attention-guided restore. This captures
first/middle/last layers at the last original query position and stores scores
before eviction. Model key/block size must match. Full-suffix fallback checks
integration but does not establish useful sparse prediction or quality.

Actual GPU gather/scatter layout, stream synchronization, alias publication/page
recycling, and generation after real eviction remain unverified locally. Save
install logs/environment freeze with results, verify artifacts on Drive, then
disconnect/delete the GPU runtime. In-memory KV does not survive runtime deletion.
No local GPU tests, installations, or benchmarks were run for this phase.
