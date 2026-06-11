# Instant4D on RTX PRO 4000 Blackwell (sm_120) — uv setup notes

This repo has been adapted to build and run under a **uv** virtualenv on an
**NVIDIA RTX PRO 4000 Blackwell** GPU. The original README targets conda +
`cu126`; that does **not** work on Blackwell.

## Environment (this machine)

| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX PRO 4000 Blackwell |
| Compute capability | **sm_120 (12.0)** |
| Driver | 580.159.04 (supports CUDA 13.0) |
| CUDA toolkit (for nvcc) | **12.8** (`/usr/local/cuda-12.8`) |
| PyTorch | **2.8.0+cu128** (ships sm_120 kernels) |
| Python | 3.11 (uv-managed venv at `.venv`) |
| OS | Ubuntu 20.04 (glibc 2.31) |

The decisive constraint: **Blackwell sm_120 needs CUDA ≥ 12.8 and PyTorch ≥ 2.7
(cu128)**. The pre-existing system CUDA 11.8 cannot compile `sm_120`.

## Quick start

```bash
just setup            # uv venv + torch 2.8.0+cu128 + python deps
just cuda-toolkit     # CUDA toolkit 12.8 (nvcc) — needs sudo, ~one-time
just download-models  # Depth-Anything + RAFT checkpoints (gdown)
just build-ext        # simple_knn + pointops2 + JIT rasterizer (sm_120)
just build-megasam    # DROID backend + lietorch (sm_120)
just install-reconstruct-deps
just pipeline panda   # reconstruct -> prune -> optimize
just test             # pytest regression suite
```

Every CUDA build needs these in the environment (the justfile sets them):
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=12.0     # Blackwell sm_120
```

## Source modifications (why each was needed)

All edits are driven by torch>2.7 / CUDA 12.8 / sm_120 / modern Pillow / glibc 2.31.

### mega-sam DROID backend + lietorch — `SLAM/mega-sam/base/`
- `.type()` → `.scalar_type()` in dispatch macros (torch>2.7 removed the implicit
  `DeprecatedTypeProperties → ScalarType` conversion):
  - `src/correlation_kernels.cu` (2×), `src/altcorr_kernel.cu` (1×)
  - `thirdparty/lietorch/lietorch/src/lietorch_gpu.cu` (19×), `lietorch_cpu.cpp` (19×)
- `base/setup.py`: added `-gencode=arch=compute_90,code=sm_90` and
  `compute_120,sm_120` to **both** extension blocks (droid_backends + lietorch).
  Without sm_120 cubin → runtime "no kernel image is available".
- Ref: kaolin#865, DPVO#100.

### diff-gaussian-rasterization — `diff-gaussian-rasterization/setup.py`
- Removed the `-g -G` (device-debug) nvcc flags, added `-O3`. `-G` disables all
  device optimization and is very slow on Blackwell. NOTE: the rasterizer is
  actually **JIT-compiled** at first import (`gaussian_renderer/diff_gaussian_rasterization.py`
  via `torch.utils.cpp_extension.load`), so it picks up `TORCH_CUDA_ARCH_LIST=12.0`
  at runtime — keep that env var set.

### simple-knn — no source change
- `#include <cfloat>` was already present (the usual FLT_MAX fix). Just build with
  `TORCH_CUDA_ARCH_LIST=12.0`.

### pointops2 — `utils/general_utils.py`
- Import path fixed to match the installed wheel layout:
  `from pointops2.functions.pointops import …` → `from pointops2.pointops import …`.

### UniDepth (xformers-free) — `SLAM/mega-sam/UniDepth/unidepth/layers/nystrom_attention.py`
- xformers has no Blackwell wheel matched to torch 2.8, and UniDepth's
  `requirements.txt` would downgrade torch to 2.2. So the hard `from xformers…`
  import is guarded and falls back to a PyTorch SDPA implementation
  (`F.scaled_dot_product_attention`), which runs natively on sm_120. At runtime
  you'll see "xFormers not available" — that's expected and fine.

### Modern-Pillow / convenience fixes
- `scene/dataset_readers.py`: `Image.fromarray(..., dtype=np.byte)` → `np.uint8`
  (Pillow 10+ rejects signed int8 for "RGB"/"RGBA").
- `script/optimize.py`, `script/prune.py`: hard-coded `Instant4D/...` paths and the
  scene name are now env-overridable (`I4D_SCENE`, `I4D_CONFIG`, `I4D_SOURCE`,
  `I4D_MODEL`, `I4D_SCENES`, `I4D_SAVE_DIR`), defaulting to repo-relative paths so
  scripts run from the repo root.
- `script/prune.py`: `make_transforms` `file_path` is now relative to the scene
  image dir (was an absolute `/data/zhanpeng/sora/...`). `filtered_cvd.npz` +
  transforms are written into `example/<scene>/` where the optimizer reads them.
- `gaussian_renderer/network_gui_websocket.py`: the viewer thread is now a daemon
  and bind failures are non-fatal; set `I4D_NO_WEBSOCKET=1` for headless runs.
- `script/optimize.py`: the save-iteration call `scene.render_evaluate_dycheck(...)`
  referenced a non-existent method → `scene.render_evaluate_sora(...)` (the actual
  method on `Scene`; panda is a `configs/sora/` scene). Without this, training
  crashes at the first `save_iterations` step (default 3000) while writing the
  novel-view wobble videos.
- `scene/__init__.py`: novel-view video codec `avc1` (H264) → `mp4v`.
  `opencv-python-headless` has no bundled H264 encoder, so `avc1` wrote 0-byte
  `.mp4` files ("Could not find encoder for codec_id=27"); `mp4v` encodes fine.

## Dependency gotchas
- **torch_scatter** (needed by DROID) must be **built from source** on Ubuntu 20.04:
  the PyG prebuilt wheel requires glibc 2.32. Use
  `uv pip install --no-build-isolation --no-binary torch_scatter torch_scatter`.
- **UniDepth** pulls `wandb` + `h5py` at import time — installed explicitly; do
  **not** install `UniDepth/requirements.txt` (it pins torch 2.2 / xformers 0.0.24).
- Depth-Anything downloads the DINOv2 ViT-L backbone from torch.hub on first run.
- UniDepthV2 weights download from HuggingFace on first run.

## Regression tests
`just test` runs `tests/`:
- `test_environment.py` — torch sees sm_120 and a kernel actually executes.
- `test_extensions.py` — simple_knn / pointops2 / rasterizer kernels run on GPU.
- `test_optimize_smoke.py` (slow) — full 4DGS optimize loop on a tiny synthetic
  Mega-SAM scene (`tests/synthetic_scene.py`), no heavy reconstruction needed.
