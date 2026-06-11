# Instant4D — uv + Blackwell(sm_120) workflow
# GPU: NVIDIA RTX PRO 4000 Blackwell (sm_120 / CC 12.0)
# Stack: CUDA toolkit 12.8 (nvcc) + PyTorch 2.8.0+cu128 + Python 3.11 (uv venv)
#
# Usage: `just` to list recipes, `just setup`, `just build-ext`, `just pipeline panda`, `just test`

set shell := ["bash", "-uc"]

# --- config ---
py        := ".venv/bin/python"
pip       := "uv pip install --python .venv"
cuda_home := "/usr/local/cuda-12.8"
arch      := "12.0"          # Blackwell sm_120
torch_idx := "https://download.pytorch.org/whl/cu128"
max_jobs  := "16"            # parallel nvcc jobs (box has 48 cores)

# Build environment exported to every CUDA compile recipe
export CUDA_HOME              := cuda_home
export TORCH_CUDA_ARCH_LIST   := arch
export MAX_JOBS               := max_jobs
export PATH                   := cuda_home + "/bin:" + env_var('HOME') + "/.local/bin:" + env_var('PATH')

# Default: show available recipes
default:
    @just --list

# ---------- environment ----------

# Show GPU / CUDA / torch info
gpu-info:
    nvidia-smi || true
    @echo "--- nvcc ---"
    {{cuda_home}}/bin/nvcc --version || echo "nvcc (12.8) not installed yet — run: just cuda-toolkit"
    @echo "--- torch ---"
    {{py}} -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda); print('device', torch.cuda.get_device_name(0)); print('capability', torch.cuda.get_device_capability(0)); print('arch_list', torch.cuda.get_arch_list())" || echo "torch not installed yet — run: just setup"

# Install CUDA toolkit 12.8 (needs sudo; provides nvcc for sm_120 builds)
cuda-toolkit:
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-8

# Create the uv virtualenv (Python 3.11)
venv:
    uv venv --python 3.11 .venv

# Install Blackwell-compatible PyTorch (cu128)
install-torch:
    {{pip}} --index-url {{torch_idx}} "torch==2.8.0" "torchvision==0.23.0"

# Install Python dependencies for the optimize / prune path
install-deps:
    {{pip}} "numpy<2.0" opencv-python-headless imagesize omegaconf plyfile tqdm \
        torchmetrics kornia timm ninja huggingface-hub websockets \
        point-cloud-utils open3d scipy matplotlib tensorboard einops imageio pytest

# Dependencies for the mega-sam reconstruct path (UniDepth/DROID/RAFT/CVD).
# torch_scatter MUST be built from source on Ubuntu 20.04 (PyG wheel needs glibc 2.32).
install-reconstruct-deps:
    {{pip}} wandb h5py
    {{pip}} --no-build-isolation --no-binary torch_scatter torch_scatter
    {{py}} -c "import torch, torch_scatter; print('torch_scatter', torch_scatter.__version__, 'OK')"

# Full environment setup (venv + torch + deps)
setup: venv install-torch install-deps
    @echo "Environment ready. Next: just cuda-toolkit (if needed) then just build-ext"

# Install gdown (uv tool) for Google Drive downloads
gdown:
    uv tool install gdown

# ---------- model checkpoints ----------

# Download pretrained checkpoints (Depth-Anything + RAFT). megasam_final.pth ships with the repo.
download-models: gdown
    mkdir -p SLAM/mega-sam/Depth-Anything/checkpoints SLAM/mega-sam/cvd_opt downloads
    test -f SLAM/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth || \
        curl -L -f -o SLAM/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth \
        https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
    test -f SLAM/mega-sam/cvd_opt/raft-things.pth || ( \
        ~/.local/bin/gdown --folder "https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT" -O downloads/raft_folder && \
        cp "$(find downloads/raft_folder -iname raft-things.pth | head -1)" SLAM/mega-sam/cvd_opt/raft-things.pth )
    @echo "Checkpoints:"; ls -la SLAM/mega-sam/checkpoints/megasam_final.pth SLAM/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth SLAM/mega-sam/cvd_opt/raft-things.pth

# ---------- CUDA extensions ----------

# Build 4DGS optimize-path extensions for sm_120 (rasterizer is JIT at import)
build-ext:
    @echo "Building simple_knn + pointops2 for sm_{{arch}} ..."
    {{pip}} --no-build-isolation ./submodule/simple-knn
    {{pip}} --no-build-isolation ./submodule/pointops2
    @echo "JIT-compiling diff_gaussian_rasterization (first import) ..."
    {{py}} -c "import gaussian_renderer.diff_gaussian_rasterization as d; print('rasterizer _C:', [f for f in dir(d._C) if 'rasterize' in f])"
    @echo "Verifying imports ..."
    {{py}} -c "import simple_knn._C, pointops2_cuda; print('optimize-path extensions OK')"

# Build fused-ssim (optional; not used by optimize.py's default SSIM path)
build-fused-ssim:
    {{pip}} --no-build-isolation ./submodule/fussed-ssim

# Build mega-sam DROID backend + lietorch (reconstruct path) for sm_120
build-megasam:
    cd SLAM/mega-sam/base && {{justfile_directory()}}/{{py}} setup.py install
    {{py}} -c "import torch, droid_backends, lietorch; from lietorch import SE3; print('droid_backends + lietorch OK')"

# ---------- pipeline ----------

# Stage 1: geometry recovery with mega-sam (depth + camera tracking + flow + CVD)
reconstruct scene="panda":
    bash script/reconstruct_uv.sh {{scene}}

# Stage 2: voxel filter / dynamic-static split -> example/<scene>/filtered_cvd.npz
prune scene="panda":
    I4D_SCENES={{scene}} {{py}} -m script.prune

# Stage 3: 4D Gaussian Splatting optimization
optimize scene="panda":
    I4D_SCENE={{scene}} {{py}} -m script.optimize

# Full pipeline for one scene
pipeline scene="panda": (reconstruct scene) (prune scene) (optimize scene)
    @echo "Pipeline complete for {{scene}}"

# ---------- tests ----------

# Run pytest regression suite
test *ARGS:
    {{py}} -m pytest -v {{ARGS}}

# Fast smoke tests only (no full training)
test-smoke:
    {{py}} -m pytest -v -m "not slow"

# ---------- maintenance ----------

# Remove build artifacts
clean:
    rm -rf diff-gaussian-rasterization/build submodule/*/build **/*.egg-info
    find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
