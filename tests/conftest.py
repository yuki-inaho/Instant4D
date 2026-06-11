"""Pytest configuration / shared fixtures for Instant4D regression tests.

These tests validate that the Instant4D stack is correctly built for this
machine's GPU (NVIDIA RTX PRO 4000 Blackwell, sm_120) under the uv environment.
"""
import os
import sys
import pathlib

import pytest

# --- make the repo importable (scene/, utils/, gaussian_renderer/, arguments/) ---
# pointops2 / simple_knn are installed as wheels (pip install ./submodule/...), so
# we only add the repo root — NOT submodule/, which would shadow the installed pkgs.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- ensure the JIT-compiled rasterizer can find nvcc + targets sm_120 ---
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
_cuda_bin = os.path.join(os.environ["CUDA_HOME"], "bin")
if _cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _cuda_bin + ":" + os.environ.get("PATH", "")


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(), reason="CUDA GPU not available")


@pytest.fixture(scope="session")
def torch_mod():
    import torch
    return torch


@pytest.fixture(scope="session")
def device(torch_mod):
    if not torch_mod.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    return torch_mod.device("cuda")
