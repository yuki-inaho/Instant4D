"""Environment regression tests.

The decisive Blackwell check: the installed PyTorch must ship kernels for the
GPU's compute capability (sm_120). If it does not, every CUDA op fails at
runtime with "no kernel image is available for execution on the device".
"""
import pytest

from conftest import requires_cuda


def test_torch_is_cu128():
    import torch
    assert torch.version.cuda is not None, "PyTorch is CPU-only; need a cu128 build"
    major, minor = (int(x) for x in torch.version.cuda.split(".")[:2])
    assert (major, minor) >= (12, 8), f"need CUDA>=12.8 for Blackwell, got {torch.version.cuda}"


@requires_cuda
def test_cuda_available():
    import torch
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1


@requires_cuda
def test_gpu_arch_supported_by_torch():
    """The current GPU's sm_XX must be in the arch list PyTorch was built for."""
    import torch
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    arch_list = torch.cuda.get_arch_list()
    assert sm in arch_list, (
        f"GPU {torch.cuda.get_device_name(0)} is {sm}, "
        f"but torch was built for {arch_list} — kernels will fail at runtime"
    )


@requires_cuda
def test_gpu_kernel_executes():
    """A real kernel must actually run on the device (not just be enumerable)."""
    import torch
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    assert torch.isfinite(c).all()
