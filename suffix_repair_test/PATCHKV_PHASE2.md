# Phase 2: offline value drift and length-conditioned planning

Implementation status: Phase 1 is preserved. Phase 2 supplies paired-prefill
collection, value-drift measurement, oracle labels, task-grouped fitting, profile
persistence, and predicted-window planning. No model has been calibrated locally.
Phase 3 now adds attention capture/selection in a separate module; see
[the Phase 3 guide](PATCHKV_PHASE3.md). Phase 4 now adds eviction-independent
checkpoints; see [the Phase 4 guide](PATCHKV_PHASE4.md). Phase 5 now provides
[reference quantized transport](PATCHKV_PHASE5.md). Real calibration and complete
end-to-end GPU validation remain outstanding.

## Algorithm and explicit choices

The implementation follows [Section IV-B, equations (6)–(9)](https://arxiv.org/html/2609.26219v1#S4.SS2):

1. A `ProfilePair` declares the exact prefix and retained token-identical suffix
   through a single-span edit. An `exact_prefill` adapter runs both complete
   contexts independently and returns sampled layers' detached CPU value tensors
   shaped `[tokens, kv_heads, head_dim]`. The Colab collector gives each prefill
   a fresh cache salt and rejects any prefix-cache hit. Values are copied before
   the next pass so CPU observations remain valid if GPU pages are recycled.
   `ProfilePair.from_contexts` computes the paper's longest prefix and maximal
   nonoverlapping suffix. The collector accepts `original_ids` plus `edited_ids`
   to use that differencing automatically. Explicit spans instead follow the
   supplied boundary, matching the existing editing API; use maximal boundaries
   consistently during calibration and recovery.
2. `observe_values` computes `1 - cosine` on aligned token values. For each token
   and layer it flattens the KV heads and dimensions. It pools token/layer
   deviations into suffix-relative blocks and uses an upper empirical quantile
   (default 0.9) as the robust aggregate. Zero/zero vectors have deviation zero;
   exactly one zero vector has deviation one. FP64 CPU arithmetic avoids FP16
   reduction overflow. Partial last blocks are included.
3. `oracle_dirty_window` finds the earliest block-start distance with the required
   fraction of stable blocks in its nonempty tail. It returns the entire suffix
   length if no such tail exists. This tests tail coverage, not merely the first
   low-drift block or an assumption of monotonic drift.
4. `fit_drift_profile` minimizes high-quantile pinball loss on log1p prefix,
   replacement, and suffix lengths plus their three pairwise interactions. It
   uses a regularized least-squares warm start and deterministic exact coordinate
   updates. Fit loss and sweep count are stored; this optimizer does not certify
   a global optimum. The paper does not prescribe a regression family/solver or
   robust aggregate, so these are explicit implementation choices.
5. Training and safety-calibration tasks must be disjoint. Each calibration
   task contributes its maximum underprediction across adjacent revisions. The
   finite-sample upper order statistic at rank `ceil((n_tasks + 1) * q)` becomes
   a nonnegative residual margin. This extra conservative calibration is an
   implementation extension; it is not proof of quality on arbitrary workloads.
   If that rank exceeds the task count, prediction falls back to full suffix
   repair. At default `q=0.95`, at least 19 independent calibration tasks are
   needed for a finite margin; many revisions of one task do not count as 19.
6. Runtime prediction adds the margin, rounds upward to a token count, and clips
   to `[0, suffix_length]`. Lengths outside the calibrated per-feature min/max
   bounds fall back to full suffix repair. A bounding box is not evidence that
   every joint length combination inside it has been observed.
7. `plan_profiled_suffix_repair` checks model identity and execution block size,
   unions optional extra repair blocks, then rounds the dirty token set into
   absolute new-context blocks through the existing Phase 1 planner. Profiling
   blocks are suffix-relative; execution blocks are absolute new-context pages.
   The boundary-page prefix is preserved, and replacement tokens remain mandatory.

`deviation_threshold` (tau), `stability_coverage` (gamma), and the aggregate
quantile are recorded, not inferred from task-answer quality. The collector
requires explicit tau and gamma. Those parameters still need model/workload
calibration on disjoint real requests before making quality or speedup claims.
No paper-specific calibrated coefficients or thresholds are invented here.

Profiles and measured observations are versioned/data-only JSON (no pickle).
Writes use temporary files and replacement so an interrupted update retains the
previous complete artifact. Profiles include model/configuration identity,
sampled layers, block size, drift settings, learned coefficients/scales, length
bounds, task split IDs/counts, quantile, residual margin, and fitting diagnostics.
Loading rejects invalid dimensions, nonfinite values, unsupported schema, and
overlapping task partitions. Input artifacts must come from a trusted matching
model configuration; model identity checking is explicit, not automatic inference
from arbitrary JSON. An optional validation dataset must contain wholly unseen
tasks and pair IDs. It is evaluated after fitting and never used to fit the margin.

## CPU checks and fitting

Use the installed virtual environment; do not install anything:

```powershell
.venv/Scripts/python.exe -m unittest discover -s suffix_repair_test -p 'test_patchkv_*.py' -v
.venv/Scripts/python.exe -m suffix_repair_test.patchkv_profile_cli --help
```

Fit real saved observations after returning from Colab; this command imports
neither torch nor the vLLM runtime:

```bash
.venv/bin/python -m suffix_repair_test.patchkv_profile_cli fit \
  --observations /path/to/observations.json \
  --validation /path/to/held-out-observations.json \
  --output-dir /path/to/new-profile-directory \
  --calibration-fraction 0.25 --quantile 0.95 --seed 0
```

Validation is optional but necessary for an independent generalization report.
Keep tasks disjoint across those two files. Splitting is by whole task, not by
revision. Results include `profile.json` and `fit_summary.json`; the summary
explicitly reports full-suffix fallback when calibration is insufficient.
Keep the original `observations.json` and `collection.json` alongside the fit.

Use a fitted profile with the existing resident-cache API:

```python
from vllm.v1.core.patchkv import plan_profiled_suffix_repair
from vllm.v1.core.patchkv_profile import DriftProfile

profile = DriftProfile.load("profile.json")
replacement_ids = llm.get_tokenizer().encode(replacement_text)
plan = plan_profiled_suffix_repair(
    len(original_ids), start, end, len(replacement_ids), block_size, profile,
    model_key=actual_model_revision_dtype_backend_key,
)
edited = llm.edit_kv_cache(
    original_ids, replacement_text, start, end, cache_salt=original_salt,
    repair_suffix_blocks=plan.repair_blocks,
)
output = llm.generate(edited, sampling_params)
```

The model key must describe the actual executing configuration and exactly match
the collection key. Do not simply override a mismatched key. This is a planner
entry point; there is not yet a new LLM API argument that automatically loads a
profile. The existing manual, direct-reuse, and full-recompute APIs are unchanged.
Original KV must still be resident; Phase 2 does not provide CPU checkpoints.

## Colab collection (GPU required, not run locally)

First follow `COLAB_PATCHKV.md` for the modified checkout, compatible installation,
preflight, and the Phase 1 smoke suite. Only after that succeeds, prepare a JSON
array with real tokenized context pairs from the chosen model's tokenizer:

```json
[
  {
    "pair_id": "task-001-revision-001",
    "task_id": "task-001",
    "original_ids": [100, 101, 102, 103, 104, 105],
    "start": 2,
    "end": 3,
    "replacement_ids": [110, 111]
  }
]
```

These example IDs describe the file format only, not a calibration workload.
Use stable task IDs for adjacent revisions. Both full contexts must be nonempty
and fit within `max_model_len` leaving space for one generated token. Choose a
small number of pairs for an initial collector smoke test. A tiny smoke collection
is not sufficient to train a useful 95th-percentile profile.

Run this shell cell in Colab, replacing `TAU` and `GAMMA` with your explicit
calibration choices (no defaults are claimed to match the paper):

```bash
%%bash
set -euo pipefail
cd /content/vllm-suffix-repair
: "${TAU:?Set the value-deviation threshold}"
: "${GAMMA:?Set the required stable-tail coverage}"
RESULT=/content/drive/MyDrive/patchkv-drift-$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python -m suffix_repair_test.patchkv_profile_cli collect-gpu \
  --pairs /content/real-pairs.json --output-dir "$RESULT" \
  --tau "$TAU" --gamma "$GAMMA" \
  2>&1 | tee /content/drive/MyDrive/patchkv-drift-collection.log
```

`collect-gpu` loads the pinned Qwen2.5-0.5B model once, runs two exact prefills per
pair, samples first/middle/last layers, and saves progress after each pair.
Configuration remains FP16, eager, in-process legacy runner, Triton attention,
32-token prefill chunks, block size 16, and 32 MiB KV cache. It rejects unsupported
cache layouts. It records inputs, effective config, source revision/state,
GPU/driver info, completion count, and failure traceback. Prefill completion is
measured only by success; no latency or quality benchmark is claimed.

GPU validation still needs to confirm the sampled value layout against the real
backend, compare measured drift with an independent reference, and exercise
profile-selected repair followed by generation. The existing smoke runner now
accepts a frozen artifact and adds a `predicted` recovery mode to each edit:

```bash
.venv/bin/python -m suffix_repair_test.patchkv_colab \
  --drift-profile /path/to/profile.json \
  --results-dir /content/drive/MyDrive/patchkv-profiled-smoke-new
```

It checks model identity/block size before model loading, records predicted token
counts and the full artifact, and verifies recomputation positions, preserved
source/prefix KV, reused values, relocated keys, and subsequent generation. If
all suffix blocks are selected, it also compares with fresh prefill. Lengths
outside the calibration domain intentionally exercise the full-repair fallback;
that is not evidence of useful sparse prediction. `--preflight-only` checks only
the environment, not the supplied profile or repair modes.

Save artifacts to Drive, verify
them, and disconnect/delete the GPU runtime after collection. CPU fitting and
profile inspection need no GPU session.

## Local validation

On 2026-11-15, the existing environment passed all 37 PatchKV CPU tests (22
preserved Phase 1 tests and 15 new profiling tests). These include synthetic
cosine/alignment checks, nonmonotonic stable-tail coverage, task isolation,
length-dependent pinball fitting, finite-sample safety fallback, artifact
round trips, planner unions, and an independent CPU fitter subprocess that
asserts neither torch nor the vLLM runtime is imported. No GPU collector or
profiled GPU recovery run has been executed. No measured model drift profile
or performance/quality result is included.
