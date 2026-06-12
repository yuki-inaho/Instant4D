#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PATCHED_FILES=(
  "SLAM/mega-sam/base/src/correlation_kernels.cu"
  "SLAM/mega-sam/base/src/altcorr_kernel.cu"
  "SLAM/mega-sam/base/thirdparty/lietorch/lietorch/src/lietorch_gpu.cu"
  "SLAM/mega-sam/base/thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp"
  "SLAM/mega-sam/base/setup.py"
)
PATCH_BACKUP_DIR="$(mktemp -d)"

for patched_file in "${PATCHED_FILES[@]}"; do
  mkdir -p "$PATCH_BACKUP_DIR/$(dirname "$patched_file")"
  cp "$patched_file" "$PATCH_BACKUP_DIR/$patched_file"
done

restore_patched_sources() {
  for patched_file in "${PATCHED_FILES[@]}"; do
    if [ -f "$PATCH_BACKUP_DIR/$patched_file" ]; then
      cp "$PATCH_BACKUP_DIR/$patched_file" "$patched_file"
    fi
  done
  rm -rf "$PATCH_BACKUP_DIR"
}
trap restore_patched_sources EXIT

eval "$(uv run python - <<'PY'
import os
import shutil
from pathlib import Path

import torch

if not os.environ.get("CUDA_HOME"):
    candidates = []
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parent.parent)
    if torch.version.cuda:
        candidates.append(Path(f"/usr/local/cuda-{torch.version.cuda}"))
    candidates.append(Path("/usr/local/cuda"))
    candidates.extend(sorted(Path("/usr/local").glob("cuda-*"), reverse=True))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "bin" / "nvcc").exists():
            print(f"export CUDA_HOME={candidate}")
            print(f"export PATH={candidate / 'bin'}:$PATH")
            break

if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"export TORCH_CUDA_ARCH_LIST='{major}.{minor}+PTX'")
PY
)"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc was not found. Install a CUDA toolkit matching torch.version.cuda or set CUDA_HOME." >&2
  exit 1
fi

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  echo "TORCH_CUDA_ARCH_LIST is unset and no CUDA GPU was detected. Set it explicitly, e.g. TORCH_CUDA_ARCH_LIST='8.9+PTX'." >&2
  exit 1
fi

uv run python - <<'PY'
from pathlib import Path

replacements = {
    "SLAM/mega-sam/base/src/correlation_kernels.cu": [
        ("volume.type()", "volume.scalar_type()"),
    ],
    "SLAM/mega-sam/base/src/altcorr_kernel.cu": [
        ("fmap1.type()", "fmap1.scalar_type()"),
    ],
    "SLAM/mega-sam/base/thirdparty/lietorch/lietorch/src/lietorch_gpu.cu": [
        ("a.type()", "a.scalar_type()"),
        ("X.type()", "X.scalar_type()"),
    ],
    "SLAM/mega-sam/base/thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp": [
        ("a.type()", "a.scalar_type()"),
        ("X.type()", "X.scalar_type()"),
    ],
}

for relpath, pairs in replacements.items():
    path = Path(relpath)
    text = path.read_text()
    updated = text
    for old, new in pairs:
        updated = updated.replace(old, new)
    if updated != text:
        path.write_text(updated)
        print(f"patched {relpath}")

setup_py = Path("SLAM/mega-sam/base/setup.py")
text = setup_py.read_text()
if "_instant4d_cuda_arch_flags" not in text:
    text = text.replace("import os.path as osp\n", "import os\nimport os.path as osp\n")
    text = text.replace(
        "\nROOT = osp.dirname(osp.abspath(__file__))\n",
        """
ROOT = osp.dirname(osp.abspath(__file__))


def _instant4d_cuda_arch_flags():
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.9+PTX")
    flags = []
    for item in arch_list.replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        with_ptx = item.endswith("+PTX")
        arch = item[:-4] if with_ptx else item
        digits = arch.replace(".", "")
        flags.append(f"-gencode=arch=compute_{digits},code=sm_{digits}")
        if with_ptx:
            flags.append(f"-gencode=arch=compute_{digits},code=compute_{digits}")
    return flags


CUDA_ARCH_FLAGS = _instant4d_cuda_arch_flags()
""",
    )
    text = text.replace(
        """                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
""",
        "                    *CUDA_ARCH_FLAGS,\n",
    )
    setup_py.write_text(text)
    print("patched SLAM/mega-sam/base/setup.py")
PY

echo "Building CUDA extensions with CUDA_HOME=${CUDA_HOME:-unset} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
uv pip install --no-build-isolation submodule/simple-knn
uv pip install --no-build-isolation submodule/pointops2
uv pip install --no-build-isolation submodule/fussed-ssim
rm -rf SLAM/mega-sam/base/build
(
  cd SLAM/mega-sam/base
  uv run python setup.py install
)
