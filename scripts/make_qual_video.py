"""Video: left = original frame, right = predicted depth. One mp4 per case.

Three cases, same as scripts/viz_qualitative.py's SCENES: static indoor,
moving person, driving. Depth is coloured on one range per clip (percentile
of the whole clip's predicted disparity), so the colour is temporally
stable instead of flickering frame to frame.

    python scripts/make_qual_video.py --out outputs/qual_video
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sokkanaem import from_checkpoint
from sokkanaem.data import build_mixed
from viz import colorize

D = "/home/hyunsu/dataset_ssd"
CASES = [
    ("static_indoor", f"tum:{D}/tum_static", ["walking_static"]),
    ("moving_person", f"bonn:{D}/bonn/rgbd_bonn_dataset",
     ["rgbd_bonn_person_tracking2"]),
    ("driving", f"kitti:{D}/kitti_zs", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/v11-longclip-spread-s0/latest.pt")
    ap.add_argument("--out", default="outputs/qual_video")
    ap.add_argument("--clip-len", type=int, default=256)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = from_checkpoint(args.ckpt, dev).eval()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gap = 6
    w = args.size * 2 + gap
    for name, spec, holdout in CASES:
        ds, _ = build_mixed([spec], clip_len=args.clip_len,
                             clip_stride=args.clip_len, size=args.size,
                             holdout=holdout, val=True)
        clip, _, _ = ds[len(ds) // 2]
        clip = clip.to(dev)

        state, depths = None, []
        with torch.no_grad():
            for t in range(clip.shape[0]):
                d, state, _ = model.step(clip[None, t], state)
                depths.append(d[0, 0].cpu().numpy())
        depths = np.stack(depths)
        disp = 1.0 / np.clip(depths, 1e-3, None)
        lo, hi = np.percentile(disp, [1, 99])

        rgb = (clip.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        path = out_dir / f"{name}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                  args.fps, (w, args.size))
        for t in range(clip.shape[0]):
            frame = np.full((args.size, w, 3), 255, np.uint8)
            frame[:, :args.size] = rgb[t]
            frame[:, args.size + gap:] = colorize(disp[t], lo, hi)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"{path}  {clip.shape[0]} frames")


if __name__ == "__main__":
    main()
