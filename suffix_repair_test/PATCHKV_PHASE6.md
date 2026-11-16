# Phase 6: local integration and Colab preparation

Local integration checks are complete; CUDA execution and real model quality
evaluation remain pending. Nothing in these scripts constitutes a measured GPU
result or a calibrated model profile until the corresponding collection runs.

## Integration audit and fixes

- Prefill now passes manager block IDs to `AttentionGroup` exactly once. The
  previous path expanded manager pages into kernel pages and the group expanded
  them again. CPU tests exercise the real metadata builder with block size 4
  and kernel block size 2, so this is a numerical integration check rather than
  a mock of the broken interface.
- Packed kernel strides and tensor parallelism are explicitly rejected. Checkpoint
  sessions validate the local worker interface. Resident edits reject active
  scheduler requests and invalid token IDs/bounds before allocation.
- Session `close()` releases snapshots, refuses active leases, and invalidates
  handles. Restore statistics are defensive deep copies. Existing transaction
  rollback, source pinning, private destination ownership, publication after
  synchronization, and unrotated values remain in place.
- Quality candidates now bind the semantic repair policy through a fingerprint
  of the drift profile and attention budget. A restored compressed checkpoint
  rejects a different bound policy; full suffix recomputation is exempt since
  its suffix is not transported. Existing unbound Phase 5 policies still load,
  but the held-out benchmark requires newly bound, measured calibration data.
  A manual-policy fingerprint binds the policy class, not individual task block
  lists; guided calibration is required for the all-mode benchmark.

The CPU causal-model harness executes actual public edit/checkpoint methods,
worker wrapper delegation, cache-copy functions, attention metadata, and repair
prefills against a tiny two-layer model. It tests direct/full/manual/drift/
attention/quantized paths, physical-page destruction, causal generation after
restoration, fresh/full numerical agreement, ownership and cleanup. Earlier
tests retain exhaustive mapping, INT4 packing, RoPE, allocation failure and
serialization coverage. Source definitions are loaded in isolation because the
minimal local environment cannot import the full vLLM runtime (NumPy is absent).
This does not establish CUDA, Triton, or full model-runner compatibility.

Local results (2026-11-15, existing Windows virtual environment): all **95 CPU
tests passed**, including the previous 80. Ruff lint and formatting, Python
compilation, and `git diff --check` passed. The original
`suffix_repair_test.test_kv_cache_edit` test could not import because NumPy is
missing; no installation was attempted. CPU-only CLI subprocess checks confirm
that workflow planning and environment inspection import neither torch nor vLLM.

Phase 6 implementation changes are in `vllm/v1/worker/kv_cache_edit.py`,
`vllm/v1/worker/patchkv_checkpoint.py`, `vllm/v1/core/kv_cache_edit.py`,
`vllm/v1/engine/core.py`, `vllm/entrypoints/patchkv.py`, and the checkpoint,
precision and transport core modules. New files are `test_patchkv_integration.py`,
`test_patchkv_workflow.py`, `patchkv_environment.py`, `patchkv_workflow.py`,
`patchkv_benchmark.py`, and this document. `patchkv_quantized_colab.py` now supports
guided calibration; the overview, Colab guide and Phase 5 notes link here.
Existing Phase 1–5 changes remain local and preserved.

## One consolidated workflow

Use the installation/overlay instructions in [COLAB_PATCHKV.md](COLAB_PATCHKV.md).
No setup command is run automatically. The source pins `torch==2.14.0` and needs
matching compiled extensions; do not assume default Colab packages match it.

CPU-only planning, including validation of supplied files:

```bash
.venv/bin/python -m suffix_repair_test.patchkv_workflow \
  --manifest /path/to/manifest.json --results-dir /path/to/new-results
```

Omit `--manifest` to plan only the minimal smoke stages. Planning imports no
CUDA runtime. After reviewing the plan, a **future Colab session** can execute:

```bash
.venv/bin/python -m suffix_repair_test.patchkv_workflow \
  --manifest /path/to/manifest.json \
  --results-dir /content/drive/MyDrive/patchkv-run-unique \
  --execute --total-timeout 1200 --stage-timeout 600
```

Execution requires `COLAB_RELEASE_TAG`. Save directly to mounted Drive so partial
results survive termination. Each stage runs in a fresh process, freeing the
model before the next stage; repeated loads trade startup overhead for clean
ownership and reliable failure isolation. Downloads and cold compilation count
toward the limits. Prepare wheels/model caches before GPU allocation. The
workflow stops on first failure/timeout and never installs, retries, or widens
tolerances. Results cannot overwrite an earlier execution summary.

Order: environment/import/FP16 preflight; resident correctness versus fresh
prefill; checkpoint creation/eviction/restore; codec/RoPE smoke; paired exact
prefill drift collection; optional independent drift validation; CPU profile
fit; guided smoke; real whole-policy precision quality collection; CPU precision
fit; held-out all-mode benchmark. Stages after smoke require real input data.

Example manifest (paths relative to the manifest):

```json
{
  "drift": {
    "pairs": "drift-pairs.json",
    "validation_pairs": "drift-validation.json",
    "tau": 0.1,
    "gamma": 0.8,
    "quantile": 0.95,
    "calibration_fraction": 0.25,
    "attention_fraction": 0.05
  },
  "precision": {
    "tasks": "precision-tasks.json",
    "candidates": "precision-grid.json",
    "calibration_tasks": ["cal-a", "cal-b"],
    "validation_tasks": ["val-a", "val-b"],
    "max_quality_loss": 0.01,
    "min_baseline_score": 0.5
  },
  "benchmark": {"tasks": "heldout-tasks.json", "repetitions": 1}
}
```

Values above illustrate configuration, not established quality thresholds. Drift
pairs use the Phase 2 schema (`pair_id`, `task_id`, `original_ids`, `edited_ids`,
or explicit edit spans). Contexts must fit 192 tokens including generation.
At quantile .95, at least 19 independent calibration tasks are needed for a finite
safety margin. Smaller datasets explicitly fall back to full suffix repair and
cannot demonstrate selective-repair speedups.

Precision and benchmark inputs are lists of tasks:

```json
[{
  "task_id": "stable-real-task-id",
  "original_ids": [1, 2, 3],
  "start": 1,
  "end": 2,
  "replacement_text": "edited content",
  "expected_answer": "actual reference answer",
  "max_tokens": 4,
  "repair_suffix_blocks": []
}]
```

Those token IDs are a schema illustration only. Supply actual tokenizer IDs and
independent tasks with meaningful answers, including insertions, deletions and
replacements. Sources are limited to 160 tokens and outputs to 1..16 tokens;
choose tasks whose answers fit that budget. Candidate grid schema and the
quality acceptance procedure are in [PATCHKV_PHASE5.md](PATCHKV_PHASE5.md).
The plan checks task leakage across drift, precision, and a third benchmark
partition. Precision calibration and validation require at least two tasks each;
these are minimum mechanics, not evidence of adequate statistical power. Failed
quality acceptance stops the workflow; do not tune against held-out answers.

## Benchmark measurements and artifacts

`patchkv_benchmark.py` loads one pinned Qwen2.5-0.5B FP16 model, performs one
unmeasured warmup, and compares checkpoint direct reuse, full suffix recompute,
manual repair, drift repair, attention repair, and quantized attention repair.
It destroys original physical pages before each restore and cycles approximate
mode order while always obtaining a full reference first. Default one repetition,
maximum three; no throughput or long-context sweeps.

Each observation saves synchronized recovery latency (including planning),
nonstreamed generation latency, their sum, GPU allocation peaks, CPU payload and
charged checkpoint bytes, actual H2D transport bytes including scales, KV RMSE
and maximum error against full recomputation, greedy continuation agreement, and
reference-answer token F1. Diagnostic KV copies are excluded from latency and
allocated-memory peaks, though their allocator reservation may remain visible.
These are small comparative measurements, **not streamed TTFT**. All modes use
the same output budget; actual generated IDs/text are saved to expose early-EOS
and output-length effects. GPU KV capacity is allocated once and reported
separately: compressed CPU checkpoints do not shrink floating-point GPU pages.
Baseline and encoded CPU checkpoints coexist; per-checkpoint charged bytes are
reported separately from diagnostic CPU tensors. Single observations are noisy.

`workflow-plan.json` records commands, input hashes, warnings and expected model
loads. `workflow-results.json`, per-stage logs, input snapshots, profile/policy
artifacts and `benchmark-results.json` persist incrementally. Child smoke scripts
also record source hashes and configuration. Process success is not a claim of
equivalent task quality: inspect numeric results and calibration evidence.

## Limitations and pending GPU validation

- Single GPU, idle synchronous in-process engine, legacy runner, eager dense
  causal attention, Triton cache layout, supported standard 1D RoPE. Packed
  strides, distributed execution, V2 runner, hybrid/sliding-window attention,
  speculative decode, LoRA, multimodal and engine concurrency remain unsupported.
- CUDA metadata/kernel behavior, attention hooks/GQA on real Qwen layers,
  eviction/restore synchronization, allocator rollback, all quantized formats,
  generation correctness and quality require Colab execution. None was run here.
- Quantization remains a correct reference path with separate dequantization,
  key-only RoPE and page placement operations. Fused kernels, pinned asynchronous
  transfer, stream overlap, clustering repaired runs, and throughput tuning are
  not implemented. Do not extrapolate paper speedups from this path.
- Attention statistics are bounded-memory CPU scoring over selected query/layer
  captures. Calibration is task-grouped and conservative; small-context model
  smoke checks do not reproduce the paper's long-context workload evaluation.
- Full-precision prefix protection is required when checkpoint tags are assigned;
  later edits requiring a larger exact prefix may be rejected. Profiles and tags
  remain frozen on recovery. There is no durable cross-process checkpoint payload
  archive; metadata JSON is not a serialized tensor checkpoint.

Run local checks using the existing environment only:

```bash
.venv/bin/python -m unittest discover -s suffix_repair_test -p 'test_patchkv_*.py'
```

On Windows use `.venv/Scripts/python.exe`. CPU validation does not authorize or
substitute for a GPU session. Do not run `--execute` locally.
