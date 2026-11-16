# Phase 5: reference quantized KV transport

Phase 5 adds real FP16, K8/V8, and packed K8/V4 checkpoint representations,
mixed frozen precision tags, reference packed H2D restoration, and whole-request
quality calibration tools. The Phase 4 source-precision baseline and all prior
repair policies remain available. Phase 6 local integration and validation
preparation is documented in [PATCHKV_PHASE6.md](PATCHKV_PHASE6.md).

**No real model quality calibration or GPU test has been run locally. No
quality-calibrated model policy is included.** CPU tests use explicitly synthetic
quality records. A passing tensor reconstruction test is not a generation-quality
certificate.

## Formats and reference codec

`vllm/v1/core/patchkv_quantization.py` implements codec
`symmetric_token_head_le_v1`. Stored source rows are `[tokens, kv_heads, K|V]`.
Each source logical block gets one precision tag shared by its layers. For T rows,
H KV heads, and head dimension D, storage including scales/padding is:

| Tag | Codes | Scales | Bytes per layer/block |
| --- | --- | --- | --- |
| FP16 | Exact FP16 K and V | None | `4*T*H*D` |
| K8/V8 | Signed INT8 K and V | FP32 K/V scale per token/head | `T*H*(2*D+8)` |
| K8/V4 | Signed INT8 K, packed signed INT4 V | Same FP32 scales | `T*H*(D+ceil(D/2)+8)` |

The scale is absmax over each token/head's dimension divided by 127 for INT8 or
7 for INT4. Quantization uses round-to-nearest-even and symmetric clipping to
`[-127,127]` or `[-7,7]`. An all-zero vector uses scale 1 and all-zero codes.
INT4 packing supports the complete signed nibble range `[-8,7]` using two's
complement, low nibble first; the symmetric quantizer leaves -8 unused. Odd D
gets a zero high padding nibble, which is validated and discarded on unpack.
Partial final blocks store valid rows only.

Codes and scales are CPU-owned tensors. `EncodedKVBlock.save/load` supports
versioned little-endian data-only JSON/base64 audit round trips, including packed
bytes and FP32 scales, with shape/size/padding validation and a load byte limit.
No pickle or NumPy is used. Full engine checkpoint disk persistence is still not
implemented: checkpoint metadata JSON alone does not contain tensor payloads.

## Risk selection and actual quality calibration

The decision structure follows [Section IV-D, equations (13)–(14)](https://arxiv.org/html/2609.26219v1#S4.SS4).
The reference implementation explicitly chooses these otherwise underspecified
details:

- Per-token value residual is squared L2 reconstruction error divided by original
  value squared norm, flattening heads/dimensions and averaging across all stored
  layers. Zero/zero has zero residual. Measurement uses FP64 reductions against
  FP32 dequantized candidate values.
- Compute `Z=sqrt(stored_normalized_attention * V8_residual)` and block risk
  `alpha*max(Z)+(1-alpha)*mean(Z)`.
- Risk >= `tau_high` selects FP16; risk >= `tau_low` selects K8/V8. Lower risk
  selects K8/V4 only when `G4` passes, otherwise K8/V8.
- `G4` compares the maximum per-token layer-mean V4 residual with
  `v4_error_limit`. This limit is **selected using measured whole-policy generation
  quality**, not justified by the residual alone. K quantization effects and
  interactions between compressed blocks are evaluated by the same quality test.

`PrecisionPolicy` has three explicit states:

- `fixed_experiment`: `experimental_fixed(model_key, block_size, format)` forces
  one suffix format for diagnostics. It is always quality-uncalibrated.
- `candidate`: an explicit risk rule for offline evaluation. It requires stored
  attention and actual model-configuration binding, but is not quality-calibrated.
- `calibrated`: requires passing real `model_execution` evidence from at least
  two calibration tasks and two disjoint held-out tasks, one consistent metric,
  fixed baselines, and model/codec/attention configuration binding. These minimum
  counts are engineering checks, not statistical confidence guarantees.

`calibrate_precision_policy` evaluates complete mixed-format restores for every
provided candidate on calibration tasks, excludes rules violating the maximum
per-task quality loss or baseline-quality floor, then selects minimum measured
payload plus an optional measured-time weight. Ties use stable policy IDs. Only
the selected rule is checked on held-out tasks; failure rejects the policy.
Default fitting requires baseline quality >=0.5 (configurable and recorded).
Synthetic evidence or missing attention binding never yields calibrated status.
The constructor/loader rejects forged calibrated status without the required
evidence fields and passing measurements. External provenance is trusted input;
JSON validation cannot verify that somebody else's claimed GPU run occurred.

Compatibility binds the model revision/configuration, cache geometry/dtype, block
size, codec, and importance collection settings. Query positions are fingerprinted
relative to the original context end, so last-query/decode-probe configurations
remain comparable across lengths. Layer names/head counts/scales and normalization
are bound. Per-source RoPE table samples are excluded from policy fingerprints
because they vary with length; full HF RoPE configuration/revision remain bound,
and Phase 4 independently validates each checkpoint's RoPE identity at recovery.
Quality evidence applies to the measured workloads, repair choices, metrics, and
sampling settings; it is not a guarantee for arbitrary new contexts.

## Checkpoint API, prefix exactness, and restoration

The legacy call without `precision_policy` remains unchanged and retains source
FP16/BF16/FP32 values. The new mixed-format path currently requires **FP16 source
and destination cache**. It never converts the active GPU attention cache to an
integer format. BF16/FP32 source caches remain supported by the legacy baseline.

```python
# Explicit diagnostic, not a calibrated policy:
from vllm.v1.core.patchkv_precision import PrecisionPolicy

policy = PrecisionPolicy.experimental_fixed(model_key, block_size, "K8/V4")
checkpoint = llm.create_kv_checkpoint(
    original_ids, model_key=model_key, cache_salt=source_salt,
    max_cpu_bytes=64 << 20, precision_policy=policy,
    protected_prefix_tokens=start, attention=importance_metadata,
)
try:
    assert llm.reset_prefix_cache()
    edited = llm.edit_kv_cache(
        original_ids, replacement_text, start, end,
        checkpoint=checkpoint, repair_suffix_blocks=selected_blocks,
    )
    output = llm.generate(edited, sampling_params)
    print(checkpoint.restore_statistics)
finally:
    checkpoint.release()
```

Protecting a prefix forces every source block intersecting it to FP16. It keeps
the unchanged prefix exact, including boundary rows. A later edit is rejected
before allocation if its prefix would include a quantized source block. Future
edit boundaries are not assumed known at original checkpoint time; callers must
protect enough prefix or retain a source-precision snapshot. This explicit
protection constraint is an additional implementation restriction.

An existing uncompressed CPU handle can create a new encoded snapshot without
reading GPU pages: `encoded = baseline.compress(policy, protected_prefix_tokens=start)`.
The new handle owns independent storage; releasing baseline does not invalidate
it. Repeated lossy encoding of an already encoded snapshot is rejected.

Creation freezes tags, policy/status, measured risks, low-bit residuals, prefix
protection, codec, and actual payload byte count in `metadata.transport_json`.
Recovering a checkpoint never reselects its precision tags. Phase 1 manual/full,
Phase 2 drift, and Phase 3 attention repair selection remain independent of this
transport decision.

Recovery reads only the exact-prefix and reused-suffix rows in the fixed repair
plan. Source-block boundaries, destination-block boundaries, and token offsets
may differ. Reads split at source block boundaries and transfer only requested
code rows and their scales/padding; discarded old edit rows and repaired suffix
rows are not transferred. Dequantization runs on the destination device, followed
by the existing shifted-key RoPE correction, then floating-point cache placement.
Values are never rotated. All reusable dependencies are placed before causal
repair prefills. Copy-on-write pages, leases, synchronization, failure rollback,
and explicit release retain the Phase 4 transaction behavior.

`restore_statistics` reports payload bytes submitted, per-format bytes, actual
checkpoint payload bytes, source-precision bytes, and repaired blocks. These are
software byte counts including scales/padding, not hardware PCIe counters or bus
transaction overhead. CPU tests exercise the same accounting without a GPU.

The bounded store reserves conservatively before encoding. Encoding scratch is
limited to one logical block across layers and its candidate representations,
not a second full-context source copy. Encoded entries add a conservative 4096-byte
ownership allowance per layer/block to their managed charge; reservations use a
larger allowance and worst-format payload. For tiny blocks, managed charge may
exceed the uncompressed baseline despite fewer tensor bytes. Payload bytes and
managed charges are therefore reported separately. Existing process-RSS and
framework allocator caveats from Phase 4 still apply.

## Colab format smoke and real quality collection

After the existing environment preflight, run one model load with three short
edit cases and the three fixed formats:

```bash
%%bash
set -euo pipefail
cd /content/vllm-suffix-repair
RESULT=/content/drive/MyDrive/patchkv-quantized-$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python -m suffix_repair_test.patchkv_quantized_colab --results-dir "$RESULT" \
  2>&1 | tee /content/drive/MyDrive/patchkv-quantized-run.log
```

The pinned small-model configuration matches prior smoke scripts. It evicts source
aliases and overwrites original physical pages before each restore, then checks
exact prefixes, dequantized reused values, independent post-dequantization RoPE,
FP16 baseline agreement, and generation. It records payload/managed memory,
submitted transfer bytes, generated outputs, continuation agreement, and a single
restore-time observation. This is not a performance benchmark or quality calibration.

For actual threshold fitting, provide `--tasks real-tasks.json --candidates grid.json`.
Each task needs a stable independent `task_id`, model-tokenized `original_ids`,
token `start/end`, `replacement_text`, nonempty `expected_answer`, optional
`repair_suffix_blocks` (default []), and `max_tokens` (1–16). Use <=160 original
tokens and total edited+generated length <=192. This lightweight collector is
restricted to short-answer tasks. It measures lowercase word-token answer F1,
not the paper's official benchmark evaluator. Repair sets and sampling stay fixed
between source-precision baseline and all candidates. Baselines, exact inputs,
outputs, code hashes, measurements, costs, and provenance are saved after each run.

The grid is a JSON list of at most eight dictionaries containing `alpha`,
`tau_low`, `tau_high`, and `v4_error_limit`. Supply explicit model/workload choices;
no calibrated defaults are provided. The runner binds each candidate to the actual
model and captured importance configuration. Choose the grid and task splits
before inspecting held-out quality; collect one example per independent task.
Multiple revisions of one task must not be split across task IDs.

Fit saved measurements on CPU, with no torch/vLLM runtime import:

```bash
.venv/bin/python -m suffix_repair_test.patchkv_precision_cli \
  --measurements /path/to/quality-measurements.json \
  --calibration-tasks train-a train-b --validation-tasks held-a held-b \
  --max-quality-loss 0.02 --min-baseline-score 0.5 \
  --output-dir /path/to/new-policy-directory
```

Output is `precision-policy.json` plus a calibration summary and input checksum.
The fixed-format smoke output cannot be passed off as risk-policy calibration.
If quality fails, keep full precision or change the grid using calibration data
and validate on new untouched tasks. Tiny passing splits do not establish broad
generalization. Copy logs/environment freeze with results, verify Drive artifacts,
and disconnect/delete the GPU runtime when finished.

## Remaining optimizations and validation

The correct reference path quantizes on CPU after full-precision D2H capture.
It transfers packed H2D codes/scales and dequantizes using ordinary device PyTorch
operations. Fused GPU quantize-gather/dequantize-scatter, representation-run
clustering, pinned asynchronous transport, and tuned kernels are not implemented.
Per-block tensor objects deliberately favor reviewability over optimal metadata
overhead. The recorded restore time includes Python, copies, and repair forwards,
with possible compilation/cold-start noise; it is not a kernel-cost benchmark.

GPU operator/layout integration, stream behavior, actual bandwidth/memory savings,
generation quality, and quality-calibrated model policies remain unverified.
All Phase 4 in-process, single-GPU, dense causal Triton/native-RoPE restrictions
still apply. See Phase 6 for the current local audit; no local GPU execution has
been performed.

## CPU validation

All 80 available PatchKV CPU tests passed on 2026-11-15: 65 existing and 15 new.
New coverage includes all INT4 nibbles/odd padding, zero values, error bounds,
packed byte counts and serialization, mixed frozen tags, attention/risk gating,
quality-held-out rejection, synthetic-status protection, codec/model/attention
compatibility, noncontiguous mappings, partial-block transfer counts, exact prefix
guarding, post-dequantization RoPE, source release/eviction, reservation failure,
and API forwarding. A subprocess fitter check verifies no torch/vLLM import.
Ruff, compilation, CLI loading, and whitespace checks also pass. These are CPU
correctness checks, not experimental model-quality or latency results.
