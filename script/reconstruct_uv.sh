#!/usr/bin/env bash
# uv-friendly Mega-SAM geometry recovery (no conda) for Instant4D on Blackwell.
# Replaces script/reconstruct.sh. Single uv venv, sm_120, CUDA 12.8.
#
# Usage:   bash script/reconstruct_uv.sh [scene]
# Env:     I4D_FRAME_LIMIT=N   process only the first N frames (fast iteration)
#          I4D_GPU=0           CUDA device to use
set -euo pipefail

SCENE="${1:-panda}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEGA="$REPO/SLAM/mega-sam"
VENV="$REPO/.venv/bin/python"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CUDA_VISIBLE_DEVICES="${I4D_GPU:-0}"

DEPTH_DIR="$MEGA/medium"
CKPT="$MEGA/checkpoints/megasam_final.pth"
ANYTHING_W="$MEGA/Depth-Anything/checkpoints/depth_anything_vitl14.pth"
RAFT_CKPT="$MEGA/cvd_opt/raft-things.pth"

# Resolve the image directory (optionally a frame-limited subset).
SRC="$REPO/example/$SCENE"
if [[ -n "${I4D_FRAME_LIMIT:-}" ]]; then
  DATA="$MEGA/medium/_subset/$SCENE"
  rm -rf "$DATA"; mkdir -p "$DATA"
  i=0
  for f in $(ls "$SRC"/*.png "$SRC"/*.jpg 2>/dev/null | sort); do
    cp "$f" "$DATA/"; i=$((i+1))
    [[ "$i" -ge "$I4D_FRAME_LIMIT" ]] && break
  done
  echo ">>> using $i-frame subset at $DATA"
else
  DATA="$SRC"
fi

cd "$MEGA"
echo "============================================================"
echo " Mega-SAM reconstruct | scene=$SCENE | data=$DATA | gpu=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

echo ">>> [1/5] UniDepth metric depth"
PYTHONPATH="$MEGA/UniDepth" "$VENV" UniDepth/scripts/demo_mega-sam.py \
  --scene-name "$SCENE" --img-path "$DATA" --outdir "$DEPTH_DIR/UniDepth/"

echo ">>> [2/5] Depth-Anything mono depth"
PYTHONPATH="$MEGA/Depth-Anything:$MEGA" "$VENV" Depth-Anything/run_videos.py \
  --encoder vitl --load-from "$ANYTHING_W" \
  --img-path "$DATA" --outdir "$DEPTH_DIR/Depth-Anything/$SCENE"

echo ">>> [3/5] DROID camera tracking"
"$VENV" camera_tracking_scripts/test_demo.py \
  --datapath "$DATA" --weights "$CKPT" --scene_name "$SCENE" \
  --mono_depth_path "$DEPTH_DIR/Depth-Anything" \
  --metric_depth_path "$DEPTH_DIR/UniDepth" --disable_vis

echo ">>> [4/5] RAFT optical flow"
"$VENV" cvd_opt/preprocess_flow.py \
  --datapath "$DATA" --model "$RAFT_CKPT" \
  --scene_name "$SCENE" --mixed_precision

echo ">>> [5/5] CVD optimization"
"$VENV" cvd_opt/cvd_opt.py --scene_name "$SCENE" --w_grad 2.0 --w_normal 5.0

echo "============================================================"
echo " DONE. Outputs:"
echo "   $MEGA/outputs_cvd/${SCENE}_sgd_cvd_hr.npz"
echo "   $MEGA/reconstructions/$SCENE/motion_prob.npy"
echo "============================================================"
