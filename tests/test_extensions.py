"""CUDA-extension regression tests.

Each test compiles/loads a custom kernel and actually runs it on the GPU, so a
failure here means the sm_120 (Blackwell) build is broken — not just missing.
"""
import numpy as np
import pytest

from conftest import requires_cuda


@requires_cuda
def test_simple_knn_distCUDA2():
    import torch
    from simple_knn._C import distCUDA2

    pts = torch.rand(1000, 3, device="cuda")
    d = distCUDA2(pts)
    torch.cuda.synchronize()
    assert d.shape == (1000,)
    assert torch.isfinite(d).all()
    assert float(d.min()) >= 0.0


@requires_cuda
def test_pointops2_knn_and_fps():
    import torch
    import pointops2_cuda  # noqa: F401  (the compiled sm_120 kernel)
    from pointops2.pointops import furthestsampling, knnquery

    n, k = 256, 8
    x = torch.rand(n, 3, device="cuda").contiguous()
    off = torch.tensor([n], dtype=torch.int32, device="cuda")
    idx, dist = knnquery(k, x, x, off, off)
    torch.cuda.synchronize()
    assert tuple(idx.shape) == (n, k)
    assert int(idx.min()) >= 0 and int(idx.max()) < n

    new_off = torch.tensor([16], dtype=torch.int32, device="cuda")
    fps_idx = furthestsampling(x, off, new_off)
    torch.cuda.synchronize()
    assert tuple(fps_idx.shape) == (16,)


@requires_cuda
def test_general_utils_knn_wrapper():
    """utils.general_utils.knn is what the rigid-loss path calls."""
    import torch
    from utils.general_utils import knn

    x = torch.rand(1, 128, 3, device="cuda")
    idx, dist = knn(x, x, k=4)
    torch.cuda.synchronize()
    assert tuple(idx.shape) == (1, 128, 4)
    assert torch.isfinite(dist).all()


@requires_cuda
def test_diff_gaussian_rasterizer_jit_builds():
    """Importing the wrapper JIT-compiles the rasterizer for the current arch."""
    import gaussian_renderer.diff_gaussian_rasterization as dgr

    assert hasattr(dgr, "GaussianRasterizationSettings")
    assert hasattr(dgr, "GaussianRasterizer")
    for fn in ("rasterize_gaussians", "rasterize_gaussians_backward", "mark_visible"):
        assert hasattr(dgr._C, fn), f"missing rasterizer entry point {fn}"
