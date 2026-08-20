"""How much of the ground truth's dynamic range does the model actually produce?

REPORT 4.32 found the model emits 0.47x the GT disparity spread on the dynamic
source, which is invisible in AbsRel alone -- a compressed field can still have
a respectable mean error. Any change aimed at that defect has to be checked
against this number, not only against accuracy, and against accuracy too: a
term that widens the range without improving AbsRel has been bought with noise.
"""
import argparse

import torch

from sokkanaem import from_checkpoint
from sokkanaem.data import build_mixed, even_subset

D = "/home/hyunsu/dataset_ssd"
SOURCES = [
    ("tum", f"tum:{D}/tum_static", ["walking_static"]),
    ("bonn", f"bonn:{D}/bonn/rgbd_bonn_dataset",
     ["rgbd_bonn_crowd2", "rgbd_bonn_person_tracking2",
      "rgbd_bonn_static_close_far"]),
]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-clips", type=int, default=40)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = from_checkpoint(args.ckpt, dev).eval()

    print(f"{'source':>8s} {'range/gt':>9s} {'std/gt':>8s} {'AbsRel':>8s}")
    for name, spec, hold in SOURCES:
        ds, _ = build_mixed([spec], clip_len=8, clip_stride=8, size=256,
                            holdout=hold, val=True)
        ld = torch.utils.data.DataLoader(even_subset(ds, args.max_clips),
                                         batch_size=1, shuffle=False)
        rr, ss, ar = [], [], []
        for ci, (clip, gt, valid) in enumerate(ld):
            if ci >= args.max_clips:
                break
            clip, gt, valid = clip.to(dev), gt.to(dev), valid.to(dev)
            d, _ = m.forward_clip(clip)
            v = valid.bool()
            if not bool(v.any()):
                continue
            s = gt[v].median() / d[v].median().clamp(min=1e-6)
            p, g = (d * s)[v], gt[v]
            ar.append(((p - g).abs() / g.clamp(min=1e-6)).mean().item())
            # spread measured in disparity, the space the compression showed up
            dp, dg = 1 / p.clamp(min=1e-3), 1 / g.clamp(min=1e-3)
            q = lambda x: (x.quantile(0.95) - x.quantile(0.05)).item()
            rr.append(q(dp) / max(q(dg), 1e-9))
            ss.append(dp.std().item() / max(dg.std().item(), 1e-9))
        f = lambda a: sum(a) / len(a)
        print(f"{name:>8s} {f(rr):9.2f} {f(ss):8.2f} {f(ar):8.4f}")


if __name__ == "__main__":
    main()
