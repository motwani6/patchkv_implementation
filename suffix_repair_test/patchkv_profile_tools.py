# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load standalone profiling code without importing the vLLM runtime."""

import importlib.util
import sys
from pathlib import Path


def load_profile_module():
    name = "patchkv_profile_standalone"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "vllm/v1/core/patchkv_profile.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_attention_module():
    name = "patchkv_attention_standalone"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "vllm/v1/core/patchkv_attention.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_checkpoint_module():
    name = "patchkv_checkpoint_standalone"
    if name not in sys.modules:
        path = (
            Path(__file__).resolve().parents[1] / "vllm/v1/core/patchkv_checkpoint.py"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_transport_support(filename):
    """Load the standalone codec or quality-policy module without vLLM imports."""
    if filename not in ("patchkv_quantization", "patchkv_precision"):
        raise ValueError("Unsupported standalone transport module.")
    name = filename + "_standalone"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "vllm/v1/core" / (filename + ".py")
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
