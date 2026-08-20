"""The sawtooth, as pictures: the frames either side of a keyframe.

Section 5.7 measures accuracy decaying between keyframes and snapping back at
the refresh. A number cannot show what the decay looks like, and the reviewer
asked for the picture. One row per frame index, columns RGB | prediction |
ground truth | relative error, on the dynamic-object source where the effect
is largest.

    python scripts/viz_sawtooth.py --ckpt work_dirs/v9-60k/latest.pt \
        --frames 24 28 29 30 31 --out outputs/sawtooth
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sokkanaem import from_checkpoint
from sokkanaem.data import build_mixed
from viz import colorize

D = "/home/hyunsu/dataset_ssd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/v9-60k/latest.pt")
    ap.add_argument("--out", default="outputs/sawtooth")
    ap.add_argument("--frames", type=int, nargs="+", default=[24, 28, 29, 30, 31])
    ap.add_argument("--clip", type=int, default=0, help="which clip to draw")
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--err-max", type=float, default=0.5)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = from_checkpoint(args.ckpt, dev).eval()
    ds, _ = build_mixed([f"bonn:{D}/bonn/rgbd_bonn_dataset"],
                        clip_len=args.clip_len, clip_stride=args.clip_len,
                        size=256, holdout=["rgbd_bonn_crowd2"], val=True)
    clip, gt, valid = ds[args.clip]
    with torch.no_grad():
        depths, masks = model.forward_clip(clip[None].to(dev))
    depths = depths[0].cpu().numpy()[:, 0]
    gtn, vn = gt.numpy()[:, 0], valid.numpy()[:, 0] > 0
    rgb = clip.numpy()

    # one colour range for every row, from the whole clip's GT: a per-frame
    # range would hide exactly the drift this figure exists to show
    gd = np.where(vn, 1.0 / np.clip(gtn, 1e-3, None), 0.0)
    lo, hi = np.percentile(gd[vn], [1, 99])

    rows = []
    for t in args.frames:
        v = vn[t]
        s = np.median(gtn[t][v]) / max(np.median(depths[t][v]), 1e-6)
        pred = depths[t] * s
        err = np.abs(pred - gtn[t]) / np.clip(gtn[t], 1e-3, None)
        absrel = float(err[v].mean())
        active = float(masks[0, t].mean())
        tiles = [(rgb[t].transpose(1, 2, 0) * 255).astype(np.uint8),
                 colorize(1.0 / np.clip(pred, 1e-3, None), lo, hi),
                 np.where(v[..., None], colorize(gd[t], lo, hi), 0),
                 np.where(v[..., None], colorize(err, 0, args.err_max), 0)]
        gap = np.full((tiles[0].shape[0], 4, 3), 255, np.uint8)
        rows.append(np.concatenate([x for tile in tiles for x in (tile, gap)][:-1],
                                   axis=1))
        print(f"frame {t:3d}  AbsRel {absrel:.4f}  active {active * 100:5.1f}%")

    vgap = np.full((4, rows[0].shape[1], 3), 255, np.uint8)
    sheet = np.concatenate([x for r in rows for x in (r, vgap)][:-1], axis=0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = out / f"sawtooth-{Path(args.ckpt).parent.name}.png"
    Image.fromarray(sheet).save(name)
    print(f"wrote {name}  (rows: {args.frames}; columns: RGB, pred, GT, error)")


if __name__ == "__main__":
    main()
