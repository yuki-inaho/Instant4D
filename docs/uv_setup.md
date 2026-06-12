# Instant4D uv setup

This branch replaces the README's conda-first setup with a uv-managed Python
3.10 environment for this workstation:

- GPU: NVIDIA RTX 4000 Ada Generation
- Compute capability: 8.9
- Driver-reported CUDA runtime: 12.4
- Local CUDA toolkit: `/usr/local/cuda-11.8`

The default environment uses PyTorch 2.7.0 CUDA 11.8 wheels so CUDA extensions
can be built with the local CUDA 11.8 `nvcc` without installing another toolkit.
`CUDA_HOME` and `TORCH_CUDA_ARCH_LIST` are still respected, so other hosts can
override the defaults.

## Create the environment

```bash
git submodule update --init --recursive
uv sync
```

Install the optional UniDepth/xFormers dependencies when running the Mega-SAM
metric depth step:

```bash
uv sync --group depth
```

## Build CUDA extensions

Instant4D needs local CUDA extensions for `simple-knn`, `pointops2`,
`fused-ssim`, and Mega-SAM's `droid_backends`/`lietorch` modules. Build them
after the uv environment is synced:

```bash
script/build_cuda_extensions.sh
```

If `CUDA_HOME` is unset, the script looks for `nvcc` on `PATH`, then for a
toolkit matching `torch.version.cuda` such as `/usr/local/cuda-11.8`, then for
other `/usr/local/cuda*` installs. If `TORCH_CUDA_ARCH_LIST` is unset, it uses
the visible GPU's compute capability, which resolves to `8.9+PTX` on this
machine.

The script also patches Mega-SAM's upstream `setup.py` at build time so its
hard-coded CUDA architecture list follows `TORCH_CUDA_ARCH_LIST`; otherwise it
would omit Ada `sm_89`.

## Run commands with uv

Run project commands through `uv run` from the repository root:

```bash
uv run python -m script.prune
uv run python -m script.optimize
```

Mega-SAM's UniDepth code is not installed as a package because its upstream
`pyproject.toml` pins older torch packages. Use `PYTHONPATH` when running its
scripts:

```bash
PYTHONPATH="$PWD/SLAM/mega-sam/UniDepth:$PYTHONPATH" \
  uv run python SLAM/mega-sam/UniDepth/scripts/demo_mega-sam.py --help
```

## Run the TVA NYX650 VGGT/GlueMap COLMAP smoke test

The local 501-frame reconstruction can be used as a COLMAP scene by linking the
source images and the VGGT/GlueMap sparse model into an ignored `data/`
directory:

```bash
mkdir -p data/tva_nyx650_0501_vggt_colmap/sparse/0
ln -sfn /home/kasm-user/Desktop/TVA_NYX650_2026_06_04_colmap_0501/images \
  data/tva_nyx650_0501_vggt_colmap/images
for name in cameras images points3D frames rigs; do
  ln -sfn /home/kasm-user/Desktop/TVA_NYX650_2026_06_04_colmap_0501/gluemap_vggt_result/gluemap_aba/$name.bin \
    data/tva_nyx650_0501_vggt_colmap/sparse/0/$name.bin
done
```

Run a short end-to-end Instant4D smoke test:

```bash
uv run python -m script.optimize \
  --config configs/local/tva_nyx650_vggt_colmap_smoke.yaml \
  --test_iterations 999999 \
  --save_iterations 999999
```

The smoke config keeps the output under
`output/tva_nyx650_0501_vggt_colmap_smoke`, uses the original JPG frames, and
limits training to 20 iterations. For longer runs, copy the config and increase
`OptimizationParams.iterations` and `position_lr_max_steps`.

## Notes

- The repository currently has no `docs/` directory on `main`; this document is
  the uv setup reference added by `rtx4000-ada-uv-setup`.
- This branch defaults to CUDA 11.8 because that matches this workstation's
  installed toolkit. For Blackwell or CUDA 12.x hosts, switch the PyTorch index
  and `torch-scatter` wheel URL in `pyproject.toml`, regenerate `uv.lock`, and
  set or install a matching CUDA toolkit.
