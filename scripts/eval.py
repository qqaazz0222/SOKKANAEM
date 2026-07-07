"""Evaluation (IDEA.md §4): accuracy + temporal stability + skip ratio.

Metrics: AbsRel, RMSE, δ<1.25 (median-scaled), static-region depth delta,
average active ratio. --sweep-tau produces the skip-accuracy trade-off
curve — the PoC Go/No-Go figure (skip 50% with AbsRel degradation <5%).

Run:
    python scripts/eval.py --ckpt sokkanaem.pt --data scannet:/data/scannet
    python scripts/eval.py --ckpt sokkanaem.pt --data scannet:/data/scannet --sweep-tau
"""
import argparse
from pathlib import Path

import torch

from sokkanaem import from_checkpoint
from sokkanaem.data import build_mixed


@torch.no_grad()
def eval_once(model, loader, dev, max_clips):
    absrel = rmse = d1 = tdelta = active = n = 0
    for ci, (clip, gt, valid) in enumerate(loader):
        if ci >= max_clips:
            break
        clip, gt, valid = clip.to(dev), gt.to(dev), valid.to(dev)
        depths, masks = model.forward_clip(clip)  # detector-driven masks
        v = valid.bool()
        # median scaling (standard for scale-ambiguous depth)
        s = gt[v].median() / depths[v].median().clamp(min=1e-6)
        p = depths * s
        absrel += ((p - gt).abs() / gt.clamp(min=1e-6))[v].mean().item()
        rmse += ((p - gt) ** 2)[v].mean().sqrt().item()
        r = torch.maximum(p / gt.clamp(min=1e-6), gt / p.clamp(min=1e-6))
        d1 += (r[v] < 1.25).float().mean().item()
        # temporal: depth change on all pixels between consecutive frames
        tdelta += (depths[:, 1:] - depths[:, :-1]).abs().mean().item()
        active += masks[:, 1:].mean().item()  # frame 0 always full
        n += 1
    return {k: v / max(n, 1) for k, v in
            dict(absrel=absrel, rmse=rmse, delta1=d1,
                 temporal_delta=tdelta, active_ratio=active).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", action="append", required=True,
                    help="dataset spec 'name:/root[:scale]', repeatable")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--max-clips", type=int, default=100)
    ap.add_argument("--sweep-tau", action="store_true",
                    help="sweep detector threshold: skip-vs-accuracy curve")
    ap.add_argument("--gmc", action="store_true",
                    help="ego-motion mode: Low-Res GMC + feature gating (§3.5)")
    ap.add_argument("--spatial-cache", action="store_true",
                    help="reuse static-patch spatial outputs (§4.5 wall-clock)")
    ap.add_argument("--holdout", action="append", default=None,
                    help="path substring of the val split (repeatable); "
                         "evaluates ONLY matching sequences")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"gmc": True, "tau_on": 0.1, "tau_off": 0.05} if args.gmc else {}
    if args.spatial_cache:
        kw["spatial_cache"] = True
    # trained [model] kwargs come from config.toml next to the ckpt
    model = from_checkpoint(args.ckpt, dev, **kw).eval()

    dataset, _ = build_mixed(args.data, clip_len=args.clip_len,
                             clip_stride=args.clip_len, size=args.size,
                             holdout=args.holdout, val=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    out = Path(args.ckpt).parent / "eval.txt"
    lines = [f"ckpt={args.ckpt} data={args.data} clips={len(dataset)} "
             f"max={args.max_clips}"]

    if not args.sweep_tau:
        taus = [(model.detector.tau_on, model.detector.tau_off)]
    elif args.gmc:  # relative-L1 feature scale (§3.5)
        taus = [(0.0, 0.0), (0.05, 0.025), (0.1, 0.05), (0.2, 0.1),
                (0.4, 0.2), (0.8, 0.4)]
    else:
        taus = [(0.0, 0.0), (0.005, 0.0025), (0.01, 0.005), (0.02, 0.01),
                (0.05, 0.025), (0.1, 0.05)]

    hdr = "tau_on   active%  AbsRel   RMSE    d1      t-delta"
    lines += [hdr, "-" * len(hdr)]
    for tau_on, tau_off in taus:
        model.detector.tau_on, model.detector.tau_off = tau_on, tau_off
        m = eval_once(model, loader, dev, args.max_clips)
        lines.append(f"{tau_on:<8g} {m['active_ratio']*100:6.1f}  "
                     f"{m['absrel']:.4f}  {m['rmse']:.4f}  {m['delta1']:.4f}  "
                     f"{m['temporal_delta']:.4f}")

    report = "\n".join(lines)
    print(report)
    with open(out, "a") as f:
        f.write(report + "\n\n")
    print(f"\nappended -> {out}")


if __name__ == "__main__":
    main()
