"""Qualitative panels: RGB | predicted depth | GT depth, one PNG per sample.

Samples are drawn from the *val* split of each dataset (sequences the model
never trained on), two per source, and the model runs in streaming mode over
the whole clip so the temporal state — and the change mask — are warmed up
exactly as they would be at deployment. The visualized frame is the last one.

Prediction is median-scaled to the GT before colouring (depth here is
scale-ambiguous, same convention as scripts/eval.py), and both maps share one
colour range taken from the GT, so pred and GT are directly comparable.

Run:
    python scripts/viz.py --ckpt work_dirs/main_v8/latest.pt --out outputs/v8
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sokkanaem import from_checkpoint
from sokkanaem.data import build_mixed

# real sequences first, then synthetic; holdout lists match configs/main_v8.toml
SOURCES = [
    ("tum", "tum:/home/hyunsu/dataset_ssd/tum_static", ["walking_static"]),
    ("bonn", "bonn:/home/hyunsu/dataset_ssd/bonn/rgbd_bonn_dataset",
     ["rgbd_bonn_crowd2", "rgbd_bonn_person_tracking2",
      "rgbd_bonn_static_close_far"]),
    ("vkitti2", "vkitti2:/home/hyunsu/dataset_ssd/vkitti2", ["Scene06"]),
    ("tartanair2", "tartanair2:/home/hyunsu/dataset_ssd/tartanair_v2",
     ["OldTownFall"]),
    ("pointodyssey", "pointodyssey:/home/hyunsu/dataset_ssd/pointodyssey",
     ["/pointodyssey/val/", "/pointodyssey/test/"]),
]

# 8 anchors of a magma-like ramp; np.interp between them is enough for a
# depth colour map and costs no dependency (matplotlib is not installed)
_ANCHORS = np.array([
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 253, 191],
], dtype=np.float32)


def colorize(x, lo, hi):
    """x: (H, W) float. Returns (H, W, 3) uint8 over the [lo, hi] range."""
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) * (len(_ANCHORS) - 1)
    i = np.floor(t).astype(int).clip(0, len(_ANCHORS) - 2)
    f = (t - i)[..., None]
    return (_ANCHORS[i] * (1 - f) + _ANCHORS[i + 1] * f).astype(np.uint8)


def panel(rgb, pred, gt, valid):
    """All inputs numpy, (3,H,W) / (H,W). Returns a PIL image of 3 tiles.

    Colour range comes from the GT's inverse depth (1st-99th percentile of
    valid pixels), and the prediction reuses it — a per-map range would make
    a wrong prediction look right."""
    v = valid > 0
    gt_disp = np.where(v, 1.0 / np.clip(gt, 1e-3, None), 0.0)
    lo, hi = np.percentile(gt_disp[v], [1, 99]) if v.any() else (0.0, 1.0)
    scale = (np.median(gt[v]) / max(np.median(pred[v]), 1e-6)) if v.any() else 1.0
    pred_disp = 1.0 / np.clip(pred * scale, 1e-3, None)
    tiles = [(rgb.transpose(1, 2, 0) * 255).astype(np.uint8),
             colorize(pred_disp, lo, hi),
             np.where(v[..., None], colorize(gt_disp, lo, hi), 0)]
    gap = np.full((tiles[0].shape[0], 4, 3), 255, np.uint8)
    return Image.fromarray(np.concatenate(
        [tiles[0], gap, tiles[1], gap, tiles[2]], axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/main_v8/latest.pt")
    ap.add_argument("--out", default="outputs/v8")
    ap.add_argument("--per-source", type=int, default=2)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--clip-len", type=int, default=8)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = from_checkpoint(args.ckpt, dev).eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for name, spec, holdout in SOURCES:
        ds, _ = build_mixed([spec], clip_len=args.clip_len,
                            clip_stride=args.clip_len, size=args.size,
                            holdout=holdout, val=True)
        # spread the picks across the split instead of taking neighbours
        idx = np.linspace(0, len(ds) - 1, args.per_source).astype(int)
        for k, i in enumerate(idx):
            clip, gt, valid = (t.unsqueeze(0).to(dev) for t in ds[int(i)])
            state, ratios = None, []
            with torch.no_grad():
                for t in range(clip.shape[1]):
                    depth, state, info = model.step(clip[:, t], state)
                    ratios.append(info["active_ratio"])
            active = float(np.mean(ratios[1:]))  # frame 0 is always a keyframe
            img = panel(clip[0, -1].cpu().numpy(), depth[0, 0].cpu().numpy(),
                        gt[0, -1, 0].cpu().numpy(), valid[0, -1, 0].cpu().numpy())
            n += 1
            path = out / f"{n:02d}_{name}_{k}_active{active*100:.0f}pct.png"
            img.save(path)
            print(f"{path}  active {active*100:5.1f}%  clip {i}/{len(ds)}")

    (out / "README.txt").write_text(
        f"ckpt: {args.ckpt}\nlayout: RGB | predicted depth | GT depth\n"
        "colour: inverse depth, range = GT 1st-99th percentile, shared by\n"
        "  prediction and GT; prediction median-scaled to GT first.\n"
        "black in the GT tile = no valid depth. active%% in the filename is\n"
        "the mean patch-activity over frames 1..N of the clip.\n")
    print(f"\n{n} panels -> {out}")


if __name__ == "__main__":
    main()
