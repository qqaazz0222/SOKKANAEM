"""What is the best score this output structure could possibly reach?

The model predicts one token per patch and upsamples. That alone caps
accuracy, independent of capacity or training. Pushing the ground truth
through the same bottleneck -- average-pool to the token grid, bilinear back
up, median-scale -- measures that cap.

The point is to tell two very different situations apart:

  model far below the cap  -> capacity or training is the bottleneck
  model near the cap       -> the patch grid is, and only finer patches or
                              higher input resolution can help

REPORT 4.17 ran this on synthetic data at patch 16 and found plenty of room.
Whether the same holds on real indoor footage was never measured, and the two
answers imply opposite next steps.

    python scripts/ceiling_probe.py --data tum:/data/tum bonn:/data/bonn
"""
import argparse

import torch
import torch.nn.functional as F

from sokkanaem.data import build_mixed
from sokkanaem.metrics import clip_scores, pooled


def through_grid(gt, valid, patch, disparity):
    """GT -> token grid -> back up, i.e. the best a patch-token head can do.

    Pools over VALID pixels only. Real depth sensors leave holes as zeros, and
    averaging those in is bad enough in depth space but catastrophic in
    disparity space, where a single zero becomes 1/eps and swamps its patch --
    the first run of this probe reported AbsRel 11.4 on TUM for exactly that
    reason.
    """
    v = valid.float()
    x = gt
    if disparity:                     # far field survives pooling much better
        x = 1.0 / x.clamp(min=1e-3)   # in inverse space
    num = F.avg_pool2d(x * v, patch)
    den = F.avg_pool2d(v, patch).clamp(min=1e-6)
    x = F.interpolate(num / den, scale_factor=patch, mode="bilinear",
                      align_corners=False)
    return 1.0 / x.clamp(min=1e-6) if disparity else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--max-clips", type=int, default=100)
    ap.add_argument("--holdout", action="append", default=None)
    ap.add_argument("--patch", type=int, nargs="+", default=[16, 8, 4])
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dataset, _ = build_mixed(args.data, clip_len=args.clip_len,
                             clip_stride=args.clip_len, size=args.size,
                             holdout=args.holdout, val=True)
    sources = [(spec.split(":")[0],
                torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False))
               for spec, ds in zip(args.data, dataset.datasets)]

    arms = [(p, d) for p in args.patch for d in (False, True)]
    print(f"{'source':>14s} {'patch':>6s} {'space':>10s} "
          f"{'AbsRel':>8s} {'RMSE':>8s} {'d1':>8s}")
    print("-" * 60)
    for name, loader in sources:
        acc = {a: [] for a in arms}
        for i, batch in enumerate(loader):
            if i >= args.max_clips:
                break
            frames, gt, valid = (t[0].to(dev) for t in batch)
            for a in arms:
                patch, disp = a
                pred = through_grid(gt, valid, patch, disp)
                # same median alignment every other number in this repo uses
                v = valid.bool()
                if not bool(v.any()):
                    continue
                s = (gt[v].median() / pred[v].median().clamp(min=1e-6))
                r = clip_scores(frames, pred * s, gt, valid)
                if r is not None:
                    acc[a].append(r)
        for a in arms:
            rows = acc[a]
            if not rows:
                continue
            m = pooled([r["_pooled"] for r in rows])
            print(f"{name:>14s} {a[0]:>6d} {'disparity' if a[1] else 'depth':>10s} "
                  f"{m['absrel']:8.4f} {m['rmse']:8.4f} {m['delta1']:8.4f}")


if __name__ == "__main__":
    main()
