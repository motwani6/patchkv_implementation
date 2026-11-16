# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only environment audit; GPU checks require the explicit --gpu flag."""

import argparse
import importlib.metadata
import inspect
import platform
import re
import sys
from pathlib import Path

from suffix_repair_test.patchkv_colab import command, save


def inspect_environment(root, gpu=False):
    required = re.search(
        r"^torch==([^\s]+)", (root / "requirements/cuda.txt").read_text(), re.M
    ).group(1)
    versions = {}
    for name in ("torch", "numpy", "transformers", "triton", "vllm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = dict(
        status="cpu_audit_only",
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        required_torch=required,
        versions=versions,
        gpu_checked=False,
        errors=[],
    )
    if not gpu:
        return report
    try:
        if sys.platform != "linux" or not (3, 11) <= sys.version_info[:2] < (3, 15):
            raise RuntimeError("Use supported Linux Colab Python 3.11–3.14.")
        if versions["torch"] is None or versions["torch"].split("+")[0] != required:
            raise RuntimeError(
                f"Source pins torch=={required}; installed {versions['torch']}. "
                "Use matching precompiled binaries."
            )
        if versions["numpy"] is None:
            raise RuntimeError(
                "NumPy is missing; the vLLM runtime cannot import. "
                "Finish environment setup before GPU work."
            )
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5):
            raise RuntimeError("Need a usable CUDA GPU with compute capability >=7.5.")
        report.update(
            gpu_checked=True,
            torch_cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            cuda_arches=torch.cuda.get_arch_list(),
            nvidia_smi=command("nvidia-smi"),
        )
        matrix = torch.ones(16, 16, device="cuda", dtype=torch.float16)
        if (matrix @ matrix)[0, 0].item() != 16:
            raise RuntimeError("FP16 CUDA matmul compatibility check failed.")
        import vllm._C  # noqa: F401

        import vllm
        from vllm import LLM
        from vllm.entrypoints.patchkv import KVCheckpointSession
        from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
        from vllm.v1.worker.patchkv_attention import CheckpointAttentionCollector
        from vllm.v1.worker.patchkv_checkpoint import restore_checkpoint_kv

        if Path(vllm.__file__).resolve().parent != root / "vllm":
            raise RuntimeError("Imported vLLM is not the modified checkout.")
        if "checkpoint" not in inspect.signature(LLM.edit_kv_cache).parameters:
            raise RuntimeError(
                "Checkpoint public API is missing from the imported vLLM."
            )
        if "causal" not in inspect.signature(build_attn_metadata).parameters:
            raise RuntimeError(
                "Attention metadata interface does not match this implementation."
            )
        report["interfaces"] = [
            cls.__name__
            for cls in (
                KVCheckpointSession,
                CheckpointAttentionCollector,
                restore_checkpoint_kv,
            )
        ]
        torch.cuda.synchronize()
        report["free_gpu_bytes"], report["total_gpu_bytes"] = torch.cuda.mem_get_info()
        report["status"] = "gpu_preflight_passed_not_model_validated"
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append(f"{type(error).__name__}: {error}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = inspect_environment(Path(__file__).resolve().parents[1], args.gpu)
    save(args.output, report)
    print(report["status"], report["errors"])
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
