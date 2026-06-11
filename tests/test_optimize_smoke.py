"""End-to-end smoke test of the 4DGS optimizer on a tiny synthetic scene.

This exercises the full optimize path on the GPU — data loading, GaussianModel
init (simple_knn), render forward/backward (diff_gaussian_rasterization), loss,
densification and the training loop — for a handful of iterations. It is the
regression guard that the sample "runs" on this Blackwell box.
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

from conftest import requires_cuda
from synthetic_scene import make_scene, write_smoke_config

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@requires_cuda
@pytest.mark.slow
def test_optimize_runs_on_synthetic_scene(tmp_path):
    scene_dir = tmp_path / "scene"
    out_dir = tmp_path / "out"
    cfg_path = tmp_path / "smoke.yaml"
    make_scene(str(scene_dir))
    write_smoke_config(str(cfg_path))

    env = dict(os.environ)
    env.update(
        I4D_NO_WEBSOCKET="1",
        I4D_CONFIG=str(cfg_path),
        I4D_SOURCE=str(scene_dir),
        I4D_MODEL=str(out_dir),
        CUDA_HOME=env.get("CUDA_HOME", "/usr/local/cuda-12.8"),
        TORCH_CUDA_ARCH_LIST=env.get("TORCH_CUDA_ARCH_LIST", "12.0"),
    )
    env["PATH"] = "/usr/local/cuda-12.8/bin:" + env.get("PATH", "")

    proc = subprocess.run(
        [sys.executable, "-m", "script.optimize"],
        cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=600,
    )
    msg = proc.stdout[-3000:] + "\n--- STDERR ---\n" + proc.stderr[-3000:]
    assert proc.returncode == 0, f"optimize.py failed:\n{msg}"
    assert "Training complete" in proc.stdout, f"did not finish training:\n{msg}"

    # outputs were written
    assert (out_dir / "cameras.json").exists()
    metrics_file = out_dir / "evaluation_metrics.json"
    assert metrics_file.exists()
    # valid JSON (no crash mid-serialization)
    json.loads(metrics_file.read_text())
