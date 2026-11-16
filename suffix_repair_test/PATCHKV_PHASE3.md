# Phase 3: attention-guided selective repair

Phase 3 implements checkpoint-time importance collection, deterministic token
selection, metadata persistence/validation, and drift-plus-attention repair
planning. Phase 1 modes and Phase 2 profiling remain available. This phase does
not itself implement CPU KV offload. Phase 4 now supplies that lifecycle;
see [the Phase 4 guide](PATCHKV_PHASE4.md).

The policy follows [Section IV-C, equations (10)–(12)](https://arxiv.org/html/2609.26219v1#S4.SS3):
collect normalized attention on the old context, select a bounded highest-score
set from the retained suffix outside the dirty token window, and block-round
the union once. Planning is complete before any reconstruction starts.

## Scoring and memory use

`vllm/v1/core/patchkv_attention.py` contains the scorer, accumulator, immutable
metadata, and selection budget. Importing metadata/selection requires only the
standard library; tensor functions import existing PyTorch lazily.

`streaming_attention_mass` consumes already-positioned query/key tensors shaped
`[selected_queries, query_heads, head_dim]` and `[keys, kv_heads, head_dim]`.
Keys represent contiguous absolute positions from zero. Every query must have
an absolute position with its own key present. Consecutive groups of query heads
map to one KV head. The scorer multiplies QK by the actual layer scale and masks
keys with positions greater than the query position, including within a chunk.

Two passes over key tiles implement stable softmax: the first finds the global
maximum/denominator per selected query/head; the second accumulates normalized
key probabilities into a token vector. Temporary logits are bounded by
`query_tile * head_tile * key_tile` (defaults 2 × 8 × 256). There is no full
context-by-context attention matrix or full-context repeated GQA key tensor.
Changing tiling can cause small floating-point differences, not changes in the
mathematical attention calculation. Invalid/nonfinite inputs fail closed.

`AttentionAccumulator` captures only configured queries and keys through the
latest configured query position. Chunk inputs are detached/copied to CPU before
buffers can be recycled or kernels can modify them. Thus GPU instrumentation
adds host copies of sampled layers' keys: storage is linear in context length
and KV-head count, plus selected queries and a token-score vector. This research
adapter is not a fused kernel or a zero-copy profiler. It computes importance on
CPU during checkpoint creation, not on the resume critical path. Full K tensors
are stored; full attention matrices are not.

Probabilities are averaged over each sampled layer's heads/queries, then summed
with equal layer weight and L1-normalized over **original checkpoint tokens**.
Generated probe keys participate in every relevant softmax denominator, but
probe token scores are excluded from the stored vector. Queries inside the
original context are also supported; tokens causally beyond all selected queries
receive zero score. No RoPE operation is performed by the scorer: both Q and K
must already have the actual model's RoPE, exactly once.

## Actual model instrumentation

`vllm/v1/worker/patchkv_attention.py` provides
`CheckpointAttentionCollector`, an opt-in context manager around **one uncached
request** at checkpoint time. It observes the attention module's real input Q/K
after RoPE (and after QK normalization where present). It uses the backend's
actual `impl.scale`, so custom attention scales are not replaced by a guessed
`1/sqrt(head_dim)`. Hooks neither modify model outputs nor write KV pages.
All hooks are removed on success or exception. Finalization rejects missing
queries/keys and frees captured Q/K buffers after producing metadata.

The adapter currently restricts execution to:

- Qwen2Attention or LlamaAttention with ordinary 1D RoPE, including partial and
  NeoX/interleaved layouts whose forward uses the supported native RoPE method.
- Triton dense causal decoder attention with unquantized KV.
- Eager execution, a single uncached token request, one cache group, matching
  execution block size, and all parallelism sizes equal to one.
- No LoRA, speculative decoding, KV transfer, multimodal embeddings, shared KV,
  sliding windows, attention sinks, ALiBi, soft-capped logits, or dual-chunk masks.

These restrictions prevent silently scoring a different attention formula.
Other layers/configurations may be supportable but are not assumed compatible.
Prefix-cache hits, nonzero initial positions, repeated source rows, packed
requests, padding, incomplete passes, wrong original tokens, and changed KV dtype
are rejected. The caller must submit only the intended request while hooks are
active. In-process execution is required by this caller-side instrumentation;
the collector does not install hooks in a remote engine process.

## Metadata and selection

`AttentionMetadata` binds scores to a model/revision/configuration key, checkpoint
ID (use the source cache salt), SHA-256 of all original token IDs, token count,
execution block size, configured query positions, sampled layer geometry/scales,
and signatures of their RoPE layout/representative table values. Revision and
complete RoPE configuration must be represented in the caller's model key;
representative table signatures alone are not an exhaustive model-identity test.

JSON save/load uses no pickle, validates schema/scoring version and score
normalization, and writes through a temporary file followed by replacement.
Compatibility checks can verify source tokens, model key, cache salt, block size,
layer specifications, and configured query positions. Different prior edited
states with the same tokens must use distinct checkpoint IDs. Scores alone
cannot certify numerical validity of an approximate source KV checkpoint.

`AttentionBudget(max_tokens=k)` or `AttentionBudget(fraction=kappa)` selects
exactly one budget type. A fraction is applied to the count of eligible tokens
in `S minus D` and rounded upward; zero selects none, one selects all. Ties prefer
lower original positions. Heap-based ranking uses O(eligible × log(budget)) time
for a small budget and O(budget) selection storage. Fixed serialized scores yield
deterministic selected tokens; bitwise score identity across devices is not
promised. Selected offsets are returned in ascending suffix order.

`plan_attention_guided_suffix_repair` validates the checkpoint/profile, predicts
the dirty window, ranks only tokens outside that window, maps their old suffix
offsets into the new context, and rounds `D union H` into absolute destination
blocks. A token outside D remains eligible even if its page intersects D; it is
not replaced with another candidate merely because rounding adds no extra page.
The token budget bounds H, not the final rounded block count. The resulting
ordinary `PatchKVPlan.repair_blocks` is passed directly to Phase 1; replacement
tokens remain mandatory and dirty blocks cannot be displaced by the budget.

## Use with an existing in-process LLM

Run this only in the supported GPU environment after the Phase 1 Colab smoke
test passes. Model/tokenizer settings must match the frozen Phase 2 profile:

```python
from uuid import uuid4
from vllm import SamplingParams
from vllm.v1.core.patchkv import plan_attention_guided_suffix_repair
from vllm.v1.core.patchkv_attention import AttentionBudget
from vllm.v1.worker.patchkv_attention import CheckpointAttentionCollector

core = llm.llm_engine.engine_core.engine_core  # In-process engine only.
runner = core.model_executor.driver_worker.model_runner
source_salt = uuid4().hex
queries = [len(original_ids) - 1, len(original_ids)]
with CheckpointAttentionCollector(
    runner, original_ids, model_key=model_key, checkpoint_id=source_salt,
    block_size=block_size, layer_names=sampled_attention_layer_names,
    query_positions=queries,
) as capture:
    llm.generate(
        {"prompt_token_ids": original_ids, "cache_salt": source_salt},
        SamplingParams(max_tokens=2, temperature=0, ignore_eos=True),
        use_tqdm=False,
    )
attention = capture.finalize()
attention.save("checkpoint-attention.json")
replacement_ids = llm.get_tokenizer().encode(replacement_text)
plan = plan_attention_guided_suffix_repair(
    original_ids, start, end, len(replacement_ids), block_size,
    profile, attention, AttentionBudget(fraction=0.05),
    model_key=model_key, checkpoint_id=source_salt,
)
edited = llm.edit_kv_cache(
    original_ids, replacement_text, start, end, cache_salt=source_salt,
    repair_suffix_blocks=plan.repair_blocks,
)
output = llm.generate(edited, sampling_params)
```

Layer names are the actual attention module names in the cache group's
`layer_names`, e.g. `model.layers.0.self_attn.attn`. The first example query is
the final prompt position; the second observes the first generated probe token.
For maximum query position q and prompt length N, generate at least
`max(1, q - N + 2)` output tokens with EOS ignored. All probes must fit within
the model context. Sampling used for probes affects importance and should be
recorded with checkpoint provenance. This resident example keeps original KV
available. Phase 4's `checkpoint=` editing argument permits recovery after the
source pages have been evicted.

## Colab validation prepared, not executed

The existing smoke runner accepts optional Phase 3 collection/planning:

```bash
.venv/bin/python -m suffix_repair_test.patchkv_colab \
  --drift-profile /path/to/profile.json --attention-fraction 0.05 \
  --attention-query-positions 82 83 \
  --results-dir /content/drive/MyDrive/patchkv-attention-smoke-new
```

The runner uses its 83-token prompt, one model load, and the same short edit
cases. It defaults to first/middle/last attention layers; use
`--attention-layers NAME ...` to configure others. It saves one attention JSON
artifact per source pass, verifies serialization compatibility, and adds an
`attention_guided` recovery mode while preserving all previous modes. The result
JSON records probe output IDs, query/layer metadata, selected suffix offsets,
budget, predicted window length, forward positions, and generation results.

If the drift profile falls back to full-suffix repair, there are no eligible
attention candidates. That validates the fallback and instrumentation but not
useful sparse selection. A matched, calibrated profile is needed to exercise H
meaningfully. No synthetic model profile is supplied as experimental evidence.

GPU validation remains necessary for actual vLLM Qwen2/Llama hook invocation,
runner/model wrappers, live GQA/RoPE tensors, and generation after guided repair.
The CPU toy-module hook tests exercise actual adapter code but do not prove CUDA
backend integration. No GPU performance/quality claim has been established.

## Paper choices and deviations

- The paper leaves normalization, exact head/layer aggregation, fractional budget
  rounding, and ties underspecified. This implementation uses equal layer means,
  L1 normalization over original tokens, ceiling budget rounding, and position
  tie-breaking. These choices are recorded/described rather than presented as
  recovered paper hyperparameters.
- The current adapter reconstructs attention probabilities from captured actual
  Q/K using bounded two-pass CPU scoring. It does not modify vLLM attention
  kernels to emit statistics and adds sampled-layer host K copies at checkpoint
  time. A fused/probability-emitting GPU collector is not implemented.
- Sampling positions can be prompt or decode probes; use decode probes for the
  paper's intended checkpoint policy. Probes participate in denominators but
  their scores are excluded from original-checkpoint metadata.
- Supported model/backend configurations are intentionally restricted as listed
  above. CPU metadata belongs to Phase 3; Phase 4 supplies CPU KV ownership and
  Phase 5 adds optional compressed transport, each documented in its own guide.

## CPU validation

```powershell
.venv/Scripts/python.exe -m unittest discover -s suffix_repair_test -p 'test_patchkv_*.py' -v
```

On 2026-11-15, all 52 tests passed: the existing 37 plus 15 Phase 3 tests. New
checks cover dense-versus-tiled references for MHA/GQA/MQA, nondefault scales,
causal masking and probe denominators, bounded logit allocation, chunked/layer
aggregation, metadata round trips and incompatibilities, deterministic ranking
and ties, dirty-boundary rounding, both edit-shift signs, and feeding the computed
repair set into the existing editor/worker. Hook tests exercise full/partial and
NeoX/interleaved RoPE, unchanged outputs, cleanup, and unsupported configurations.
All tensor checks ran on CPU. No dependencies were installed and no GPU tests
were run for Phase 3. Phase 4's separate checkpoint tests are documented in its guide.
