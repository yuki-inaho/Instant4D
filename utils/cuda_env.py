import os
import shutil
from pathlib import Path

import torch


def _cuda_version_suffix() -> str | None:
    if torch.version.cuda:
        return torch.version.cuda
    return None


def _candidate_cuda_homes() -> list[Path]:
    candidates: list[Path] = []

    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parent.parent)

    cuda_version = _cuda_version_suffix()
    if cuda_version:
        candidates.append(Path(f"/usr/local/cuda-{cuda_version}"))

    candidates.append(Path("/usr/local/cuda"))
    candidates.extend(sorted(Path("/usr/local").glob("cuda-*"), reverse=True))

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def configure_cuda_extension_build() -> None:
    """Best-effort defaults for PyTorch CUDA extension builds.

    Explicit CUDA_HOME and TORCH_CUDA_ARCH_LIST values always win. If unset, use
    the CUDA toolkit matching the installed torch wheel and the current GPU's
    compute capability. This keeps local setup convenient without baking one
    workstation's paths into the project.
    """

    if "CUDA_HOME" not in os.environ:
        for cuda_home in _candidate_cuda_homes():
            nvcc = cuda_home / "bin" / "nvcc"
            if nvcc.exists():
                os.environ["CUDA_HOME"] = str(cuda_home)
                os.environ["PATH"] = str(nvcc.parent) + os.pathsep + os.environ.get("PATH", "")
                break

    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}+PTX"
