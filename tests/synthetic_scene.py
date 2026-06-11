"""Generate a tiny synthetic Mega-SAM-format scene for fast regression tests.

The 4DGS optimizer (`script/optimize.py`) consumes, per scene directory:
  - transforms_train.json / transforms_test.json  (cameras, intrinsics)
  - <frame>.png images referenced by each frame's file_path
  - filtered_cvd.npz  (point cloud: xyz, rgb, prob_motion, time_stamp, scale_time)

This module writes all of the above for a handful of points / frames so the
whole optimize path (GaussianModel -> render -> loss -> backward -> densify)
can be exercised on the GPU in a couple of seconds, without running the heavy
mega-sam reconstruction pipeline.
"""
import json
import os

import numpy as np
from PIL import Image


def _look_at_c2w(eye, target, world_up=(0.0, 1.0, 0.0)):
    """OpenCV/COLMAP convention camera-to-world (x right, y down, z forward)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    z = target - eye
    z /= np.linalg.norm(z) + 1e-12          # forward = camera +Z
    x = np.cross(np.asarray(world_up, dtype=np.float64), z)
    x /= np.linalg.norm(x) + 1e-12          # right = camera +X
    y = np.cross(z, x)                        # down  = camera +Y
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye
    return c2w


def make_scene(path, n_points=400, n_frames=6, H=90, W=120,
               time_duration=(0.0, 3.0), radius=3.0, seed=0):
    """Write a synthetic scene under `path`. Returns the directory path."""
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(seed)
    t0, t1 = time_duration

    # --- point cloud: a small blob near the origin ---
    xyz = (rng.standard_normal((n_points, 3)) * 0.25).astype(np.float32)
    rgb = rng.uniform(0.1, 0.9, size=(n_points, 3)).astype(np.float32)
    prob_motion = rng.uniform(0.0, 1.0, size=(n_points,)).astype(np.float32)
    time_stamp = rng.uniform(t0, t1, size=(n_points,)).astype(np.float32)
    scale_time = np.full((n_points,), (t1 - t0), dtype=np.float32)
    np.savez(
        os.path.join(path, "filtered_cvd.npz"),
        xyz=xyz, rgb=rgb, prob_motion=prob_motion,
        time_stamp=time_stamp, scale_time=scale_time,
        intrinsic=np.eye(3, dtype=np.float32),
        cam_c2w=np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1)),
    )

    # --- cameras on an arc looking at the origin ---
    fl = float(W)                       # ~53 deg horizontal FoV
    cx, cy = W / 2.0, H / 2.0
    frames = []
    times = np.linspace(t0, t1, n_frames)
    for i in range(n_frames):
        ang = (i / max(n_frames, 1)) * (np.pi / 3) - np.pi / 6   # -30..+30 deg
        eye = (radius * np.sin(ang), 0.4, -radius * np.cos(ang))
        c2w = _look_at_c2w(eye, target=(0.0, 0.0, 0.0))
        name = f"{i + 1:05d}"
        # a deterministic non-trivial image as ground truth
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[..., 0] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
        img[..., 1] = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
        img[..., 2] = (40 + 30 * i) % 256
        Image.fromarray(img, "RGB").save(os.path.join(path, name + ".png"))
        frames.append({
            "file_path": name,
            "transform_matrix": c2w.tolist(),
            "time": float(times[i]),
        })

    meta = {"w": W, "h": H, "fl_x": fl, "fl_y": fl, "cx": cx, "cy": cy,
            "frames": frames}
    for fn in ("transforms_train.json", "transforms_test.json"):
        with open(os.path.join(path, fn), "w") as f:
            json.dump(meta, f, indent=2)
    return path


SMOKE_CONFIG = {
    "gaussian_dim": 4,
    "time_duration": [0.0, 3.0],
    "num_pts": 400,
    "num_pts_ratio": 1.0,
    "rot_4d": True,
    "force_sh_3d": False,
    "batch_size": 1,
    "exhaust_test": True,
    "ModelParams": {
        "sh_degree": 0, "source_path": "", "model_path": "", "images": "images",
        "resolution": 1, "white_background": False, "data_device": "cuda",
        "eval": True, "extension": ".png", "num_extra_pts": 0, "loaded_pth": "",
        "frame_ratio": 1, "dataloader": False,
    },
    "PipelineParams": {
        "convert_SHs_python": False, "compute_cov3D_python": False, "debug": False,
        "env_map_res": 0, "env_optimize_until": 1000000000, "env_optimize_from": 0,
        "eval_shfs_4d": True,
    },
    "OptimizationParams": {
        "iterations": 8, "position_lr_init": 0.00016, "position_t_lr_init": -1.0,
        "position_lr_final": 0.0000016, "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": 8, "feature_lr": 0.0025, "opacity_lr": 0.05,
        "scaling_lr": 0.005, "rotation_lr": 0.001, "percent_dense": 0.01,
        "lambda_dssim": 0.2, "thresh_opa_prune": 0.005, "densification_interval": 2,
        "opacity_reset_interval": 100000, "densify_from_iter": 2,
        "densify_until_iter": 6, "densify_grad_threshold": 0.0002,
        "densify_grad_t_threshold": 0.0002 / 40, "densify_until_num_points": -1,
        "final_prune_from_iter": -1, "sh_increase_interval": 1000,
        "lambda_opa_mask": 0.0, "lambda_rigid": 0.0, "lambda_motion": 0.0,
    },
}


def write_smoke_config(path):
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(SMOKE_CONFIG, f)
    return path


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/i4d_smoke/scene"
    make_scene(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cfg = os.path.join(os.path.dirname(out), "smoke.yaml")
    try:
        write_smoke_config(cfg)
    except Exception as e:
        print("config write skipped:", e)
    print("wrote synthetic scene to", out)
