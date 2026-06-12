import csv
import json
import os
import random
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from script.optimize import merge_config_into_args, provided_cli_dests
from utils.general_utils import safe_state
from utils.image_utils import psnr


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def sorted_cameras(scene, split):
    train = [(camera, "train") for camera in scene.train_cameras[1.0]]
    test = [(camera, "test") for camera in scene.test_cameras[1.0]]
    if split == "train":
        cameras = train
    elif split == "test":
        cameras = test
    else:
        cameras = train + test
    return sorted(cameras, key=lambda item: (item[0].timestamp, item[0].image_name))


def tensor_to_bgr(image):
    array = image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    array = (array * 255).astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def write_video(video_path, frames, fps):
    if not frames:
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def main():
    parser = ArgumentParser(description="Render a checkpoint along the loaded COLMAP camera trajectory")
    lp = ModelParams(parser)
    OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", default="", type=str)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--output_path", default="", type=str)
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--max_frames", default=-1, type=int)
    parser.add_argument("--save_gt", action="store_true")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--seed", type=int, default=6666)

    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    cli_values = vars(args).copy()
    cli_provided = provided_cli_dests(parser, argv)

    cfg_path = args.config or "configs/sora/panda.yaml"
    args.config = cfg_path
    cfg = OmegaConf.load(args.config)
    merge_config_into_args(args, cfg)
    args.config = cfg_path

    for key in cli_provided:
        if key != "config":
            setattr(args, key, cli_values[key])

    setup_seed(args.seed)
    safe_state(args.quiet)

    lp_ = lp.extract(args)
    pp_ = pp.extract(args)

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        if args.iteration < 0:
            raise ValueError("Provide --checkpoint or --iteration")
        checkpoint_path = os.path.join(lp_.model_path, f"chkpnt{args.iteration}.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    output_path = args.output_path
    if not output_path:
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_path = os.path.join(lp_.model_path, f"{checkpoint_name}_trajectory")

    render_dir = os.path.join(output_path, "renders")
    gt_dir = os.path.join(output_path, "gt")
    os.makedirs(render_dir, exist_ok=True)
    if args.save_gt:
        os.makedirs(gt_dir, exist_ok=True)

    background = torch.tensor(
        [1, 1, 1] if lp_.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    gaussians = GaussianModel(
        lp_.sh_degree,
        gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration,
        rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d,
        sh_degree_t=2 if pp_.eval_shfs_4d else 0,
    )
    scene = Scene(
        lp_,
        gaussians,
        shuffle=False,
        num_pts=args.num_pts,
        num_pts_ratio=args.num_pts_ratio,
        time_duration=args.time_duration,
    )

    model_params, loaded_iteration = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    gaussians.restore(model_params, None)

    cameras = sorted_cameras(scene, args.split)
    if args.max_frames > 0:
        cameras = cameras[: args.max_frames]

    rows = []
    video_frames = []
    with torch.no_grad():
        for index, (camera, split_name) in enumerate(cameras):
            viewpoint = camera.cuda()
            gt_image = viewpoint.image.clamp(0, 1)
            rendered = render(viewpoint, gaussians, pp_, background)["render"].clamp(0, 1)
            frame_psnr = psnr(rendered[None], gt_image[None]).item()

            file_stem = f"{index:05d}_{camera.image_name}"
            render_path = os.path.join(render_dir, f"{file_stem}.png")
            save_image(rendered, render_path)
            if args.save_gt:
                save_image(gt_image, os.path.join(gt_dir, f"{file_stem}.png"))
            if not args.no_video:
                video_frames.append(tensor_to_bgr(rendered))

            rows.append(
                {
                    "index": index,
                    "image_name": camera.image_name,
                    "split": split_name,
                    "timestamp": float(camera.timestamp),
                    "psnr": frame_psnr,
                    "render_path": render_path,
                }
            )

    if not args.no_video:
        write_video(os.path.join(output_path, "reconstruction.mp4"), video_frames, args.fps)

    metrics_path = os.path.join(output_path, "metrics.csv")
    with open(metrics_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["index", "image_name", "split", "timestamp", "psnr", "render_path"])
        writer.writeheader()
        writer.writerows(rows)

    psnr_values = np.array([row["psnr"] for row in rows], dtype=np.float64)
    summary = {
        "checkpoint_path": checkpoint_path,
        "loaded_iteration": int(loaded_iteration),
        "split": args.split,
        "num_frames": len(rows),
        "mean_psnr": float(psnr_values.mean()) if len(psnr_values) else None,
        "median_psnr": float(np.median(psnr_values)) if len(psnr_values) else None,
        "min_psnr": float(psnr_values.min()) if len(psnr_values) else None,
        "max_psnr": float(psnr_values.max()) if len(psnr_values) else None,
        "metrics_path": metrics_path,
        "render_dir": render_dir,
    }
    with open(os.path.join(output_path, "metrics_summary.json"), "w") as json_file:
        json.dump(summary, json_file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
