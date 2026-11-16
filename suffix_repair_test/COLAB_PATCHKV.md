# Colab PatchKV smoke validation

For the consolidated Phase 1–5 validation, calibration and small held-out
benchmark workflow, see [PATCHKV_PHASE6.md](PATCHKV_PHASE6.md). The setup below
still applies; the consolidated script defaults to planning without GPU use.

No GPU results have been collected by these files. This is a correctness smoke
test, not a latency benchmark. Use one Colab session and stop on the first error.

## Compatibility review before allocating GPU time

- This checkout pins `torch==2.14.0` in both `pyproject.toml` and
  `requirements/cuda.txt`. A matching published precompiled wheel and its CUDA/
  PyTorch dependencies must be available. Do not substitute arbitrary stable
  vLLM binaries or bypass dependency resolution with `--no-deps`. If wheel lookup
  or dependency resolution fails, stop and review the install log; do not spend
  the session compiling vLLM from source.
- The local CUDA install guide requires compute capability >= 7.5 and describes
  CUDA 12.9 binaries. T4 is a candidate, not a validated target for this branch.
  `nvidia-smi` reports driver capability, not the installed PyTorch CUDA runtime.
  The preflight exercises FP16 CUDA matmul and imports `vllm._C`; actual kernel
  compatibility is established only by running the model smoke test.
- Use `TRITON_ATTN`, FP16, block size 16, single GPU, in-process engine, legacy
  runner, eager execution, and prefix caching. Automatic backend selection may
  choose a cache layout the editor cannot handle. BF16 is unsuitable on T4.
  This script does not validate the V2 runner or multiprocess RPC serialization.
- The editor supports resident dense causal KV with standard 1D RoPE only:
  no sliding-window/hybrid attention, quantized KV, offload, LoRA, speculative
  decoding, multimodal inputs, or distributed execution in this test.
- Qwen2.5-0.5B-Instruct has a small dense Qwen2 architecture and full NeoX RoPE.
  The script pins model revision `c89bee90d9f811437d9735454613c35b4a3c4dc8`.
  Its independent relocation oracle deliberately rejects other RoPE layouts.
- Full repair is compared with an uncached fresh prefill with FP16 tolerances
  `atol=rtol=0.03`, recording maximum absolute error. Different prefill shapes
  need not be bitwise equal. Sparse repair is approximate; generated text is
  recorded, but quality equivalence is not claimed. Do not widen tolerances to
  hide a failure without inspecting the failing layer and error.

The existing worker RPC synchronizes the device before and after editing, pins
source pages, and publishes destination pages after successful execution.
Selective repair seeds reused dependencies before causal prefills. The Phase 6
CPU audit fixed duplicate manager/kernel block mapping and added lifecycle and
policy compatibility checks; CUDA behavior remains unvalidated.

## Transfer this uncommitted implementation

Do this locally using the existing environment (no install, commit, or push):

```powershell
.venv/Scripts/python.exe suffix_repair_test/prepare_colab_overlay.py "$env:TEMP/patchkv-overlay.tar.gz"
```

This packages modified tracked files and nonignored untracked files over HEAD,
including these scripts, with a base-commit manifest. Inspect the printed file
list before uploading. It leaves the working tree and index unchanged. Preserve
the base checkout separately if its HEAD cannot be fetched from upstream.

## Colab cells

First mount Drive and upload the overlay using Colab's file sidebar. Prepare the
checkout and download dependencies/model on a CPU runtime where practical.
Switching to a GPU runtime can reset `/content`; keep the overlay and downloaded
caches on Drive and repeat checkout/environment setup if necessary. A venv in
`/content` may be faster and avoids relying on a venv surviving a runtime switch.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Use a full Git checkout, retaining `.git` for precompiled-wheel discovery. The
following shell cell assumes `/content/patchkv-overlay.tar.gz` was uploaded and
`uv` is already available. It does not install Python or uv. If uv is absent,
stop and arrange the environment before spending GPU time.

```bash
%%bash
set -euo pipefail
command -v uv
test ! -e /content/vllm-suffix-repair
BASE=$(tar -xOf /content/patchkv-overlay.tar.gz patchkv_overlay.json | \
  uv run --no-project --offline --no-python-downloads python -c \
  'import json,sys; print(json.load(sys.stdin)["base_commit"])')
git clone --filter=blob:none --no-checkout https://github.com/jimma2-del/vllm-suffix-repair.git /content/vllm-suffix-repair
cd /content/vllm-suffix-repair
git checkout --detach "$BASE"
tar -xzf /content/patchkv-overlay.tar.gz
uv venv --python 3.12 --no-python-downloads
test -x .venv/bin/python
.venv/bin/python -c 'import json; m=json.load(open("patchkv_overlay.json")); assert not m["deleted"], "Overlay contains deletions: apply them explicitly before proceeding"'
git fetch https://github.com/vllm-project/vllm.git main
WHEEL_BASE=$(git merge-base HEAD FETCH_HEAD)
VLLM_PRECOMPILED_WHEEL_COMMIT="$WHEEL_BASE" VLLM_USE_PRECOMPILED=1 \
  uv pip install --python .venv/bin/python -e . --torch-backend=auto \
  2>&1 | tee /content/drive/MyDrive/patchkv-install.log
uv pip freeze --python .venv/bin/python > /content/drive/MyDrive/patchkv-freeze.txt
```

Python 3.12 must already exist for this cell. If it does not, use an already
available version within this checkout's supported range (3.11–3.14). Both
`--no-python-downloads` flags prevent uv from provisioning Python. If HEAD is
not fetchable from the project fork, use a transferred full checkout instead of
cloning; do not silently test a different base. This project has committed editor
changes in its history, so cloning upstream vLLM alone is insufficient. The
installation explicitly selects the upstream merge-base for binary lookup rather
than allowing a fallback to unrelated nightly binaries. An explicitly reviewed matching wheel
can be selected with `VLLM_PRECOMPILED_WHEEL_LOCATION` when needed; no wheel is
assumed compatible in advance. See the upstream
[editable precompiled installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#python-only-build).

Select a GPU runtime. Run preflight once, before downloading/loading the model:

```bash
%%bash
set -euo pipefail
cd /content/vllm-suffix-repair
RESULT=/content/drive/MyDrive/patchkv-preflight-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RESULT"
.venv/bin/python -m suffix_repair_test.patchkv_colab \
  --preflight-only --results-dir "$RESULT" 2>&1 | tee "$RESULT/run.log"
```

Then run the smoke suite once. Each of the three edits uses an 83-token source,
four recovery modes, and two generated tokens. One model load serves all cases.
The 32-token prefill budget exercises chunk boundaries. Separated selected blocks
exercise noncontiguous repair; insertions/deletions test both RoPE shift signs.

```bash
%%bash
set -euo pipefail
cd /content/vllm-suffix-repair
RESULT=/content/drive/MyDrive/patchkv-smoke-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RESULT"
cp /content/patchkv-overlay.tar.gz "$RESULT/"
cp /content/drive/MyDrive/patchkv-{install.log,freeze.txt} "$RESULT/"
.venv/bin/python -m suffix_repair_test.patchkv_colab \
  --results-dir "$RESULT" 2>&1 | tee "$RESULT/run.log"
```

`results.json` is updated atomically after each mode, and includes configuration,
GPU/driver information, source hashes, exact inputs, forward positions, numerical
errors, generated output, status, and a traceback on failure. A hard runtime kill
can leave status `running`; completed modes remain reviewable. `run.log` preserves
startup/backend diagnostics. Keep the overlay, install log, and freeze file with
the results. Verify that all three cases and all 12 modes say `passed`, then
download a backup from Drive before disconnecting and deleting the GPU runtime.
Engine shutdown releases worker resources; it does not end Colab GPU billing.

Do not run the older larger regression or timing sweeps until this smoke suite
passes. Subsequent work should validate chained edits, the V2 runner, and a
separate multiprocess session, then assess quality before benchmarking.
