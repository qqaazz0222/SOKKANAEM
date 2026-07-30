"""Is the depth-bin head actually the far-range RMSE bottleneck? (PLAN T1-5)

Cheap decision aid before burning 3 x 1.7 h on bin ablations. Two numbers per
GT-depth decile:

  model   what the checkpoint predicts
  ceiling what the SAME bin centres could predict at best — GT snapped to its
          own two bracketing centres with the exact linear interpolation
          bin_ce_loss supervises, i.e. the quantization floor of the head

If ceiling << model in the deciles that carry the RMSE, bins are not the
bottleneck and T1-5 is not worth running.

    python scripts/bin_probe.py --ckpt work_dirs/v8-teacherfree-60k/latest.pt
"""
import argparse

import torch

from sokkanaem import checkpoint_config, from_checkpoint
from sokkanaem.data import build_mixed


def snap(gt, centres):
    """GT depth -> the best depth the bin expectation can represent."""
    lg = gt.clamp(min=1e-3).log()
    hi = torch.searchsorted(centres, lg.flatten().contiguous()).clamp(1, len(centres) - 1)
    lo = hi - 1
    t = ((lg.flatten() - centres[lo]) / (centres[hi] - centres[lo])).clamp(0, 1)
    # the head's output is the expectation over centres, so the best it can do
    # with mass on the two bracketing bins is their t-weighted mean
    return ((1 - t) * centres[lo] + t * centres[hi]).view_as(gt).exp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/v8-teacherfree-60k/latest.pt")
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--holdout", action="append", default=[])
    ap.add_argument("--clips", type=int, default=30)
    ap.add_argument("--edges", default="0,5,10,20,40,80,1e9")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = from_checkpoint(args.ckpt, device=dev).eval()
    size = checkpoint_config(args.ckpt).get("args", {}).get("size", 256)
    centres = model.decoder.bin_centres().detach()
    ds, _ = build_mixed(args.data, clip_len=8, clip_stride=8, size=size,
                        holdout=args.holdout, val=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    edges = [float(x) for x in args.edges.split(",")]

    # sums per decile: [n, abs_rel, sq_err] for model and for the ceiling
    acc = torch.zeros(len(edges) - 1, 5, dtype=torch.float64)
    with torch.no_grad():
        for ci, (clip, gt, valid) in enumerate(loader):
            if ci >= args.clips:
                break
            clip, gt, valid = clip.to(dev), gt.to(dev), valid.to(dev)
            state, preds = None, []
            for t in range(clip.shape[1]):
                d, state, _ = model.step(clip[:, t], state)
                preds.append(d)
            p = torch.stack(preds, 1)
            # same median scaling as eval.py, one factor per clip
            v = valid > 0.5
            p = p * (gt[v].median() / p[v].median())
            c = snap(gt, centres)
            for i in range(len(edges) - 1):
                m = v & (gt >= edges[i]) & (gt < edges[i + 1])
                if not bool(m.any()):
                    continue
                g = gt[m]
                acc[i, 0] += m.sum().item()
                acc[i, 1] += ((p[m] - g).abs() / g).sum().item()
                acc[i, 2] += ((p[m] - g) ** 2).sum().item()
                acc[i, 3] += ((c[m] - g).abs() / g).sum().item()
                acc[i, 4] += ((c[m] - g) ** 2).sum().item()

    print(f"{args.ckpt}  bins={len(centres)}  "
          f"range={centres[0].exp():.2f}-{centres[-1].exp():.1f} m")
    print(f"{'depth (m)':>14} {'px%':>6} {'AbsRel':>8} {'ceil':>8} "
          f"{'RMSE':>9} {'ceil':>9} {'RMSE share':>11}")
    tot_n, tot_sq = acc[:, 0].sum(), acc[:, 2].sum()
    for i in range(len(edges) - 1):
        n = acc[i, 0]
        if n == 0:
            continue
        print(f"{edges[i]:6.0f}-{edges[i+1]:<7.0f} {100*n/tot_n:6.1f} "
              f"{acc[i,1]/n:8.4f} {acc[i,3]/n:8.4f} "
              f"{(acc[i,2]/n).sqrt():9.3f} {(acc[i,4]/n).sqrt():9.3f} "
              f"{100*acc[i,2]/tot_sq:10.1f}%")
    print(f"{'ALL':>14} {100.0:6.1f} {acc[:,1].sum()/tot_n:8.4f} "
          f"{acc[:,3].sum()/tot_n:8.4f} {(tot_sq/tot_n).sqrt():9.3f} "
          f"{(acc[:,4].sum()/tot_n).sqrt():9.3f}")


if __name__ == "__main__":
    main()
