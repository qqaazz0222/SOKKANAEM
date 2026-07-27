"""Evaluation (IDEA.md §4): accuracy + temporal stability + skip ratio.

Metrics: AbsRel, RMSE, δ<1.25 (median-scaled), static-region depth delta,
average active ratio. --sweep-tau produces the skip-accuracy trade-off
curve — the PoC Go/No-Go figure (skip 50% with AbsRel degradation <5%).

Run:
    python scripts/eval.py --ckpt sokkanaem.pt --data scannet:/data/scannet
    python scripts/eval.py --ckpt sokkanaem.pt --data scannet:/data/scannet --sweep-tau
"""
import argparse
import json
from pathlib import Path

import torch

from sokkanaem import from_checkpoint
from sokkanaem.data import build_mixed
from sokkanaem.metrics import clip_scores, pooled


@torch.no_grad()
def eval_once(model, loader, dev, max_clips, constant=False):
    """constant=True replaces the prediction with the clip's best possible
    constant depth — the degenerate control. It scores t-delta 0 and OPW 0
    while being useless, which is exactly why TCE is reported (metrics.py)."""
    acc = {k: [] for k in ("absrel", "rmse", "delta1", "temporal_delta",
                           "opw", "tce", "active_ratio", "_pooled")}
    skipped = 0
    for ci, (clip, gt, valid) in enumerate(loader):
        if ci >= max_clips:
            break
        clip, gt, valid = clip.to(dev), gt.to(dev), valid.to(dev)
        depths, masks = model.forward_clip(clip)  # detector-driven masks
        v = valid.bool()
        if constant:
            depths = torch.full_like(gt, gt[v].median())
        # median scaling (standard for scale-ambiguous depth)
        s = gt[v].median() / depths[v].median().clamp(min=1e-6)
        p = depths * s
        sc = clip_scores(clip[0], p[0], gt[0], valid[0])
        if sc is None:  # clip has no valid GT pixel at all — see clip_scores
            skipped += 1
            continue
        for k, val in sc.items():
            acc[k].append(val)
        acc["active_ratio"].append(masks[:, 1:].mean().item())  # frame 0 full
    # pixel-pooled dataset-level metrics are the primary numbers (convention,
    # and not hostage to a few blown-up clips); per-clip std reports spread
    out = dict(pooled(acc.pop("_pooled")))
    t = {k: torch.tensor(v) for k, v in acc.items()}
    out["active_ratio"] = t["active_ratio"].mean().item()
    out["n"] = len(t["absrel"])
    out["skipped"] = skipped
    out["absrel_std"] = t["absrel"].std().item() if out["n"] > 1 else 0.0
    out["absrel_clip"] = t["absrel"].mean().item()
    out["per_clip"] = {k: v.tolist() for k, v in t.items()}
    return out


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
    ap.add_argument("--gate-mode", default="delta", choices=["delta", "drop"],
                    help="§4.4 gating-position ablation: Δ-gating vs token drop")
    ap.add_argument("--scores-tag", default=None,
                    help="name for the per-clip JSON dump (default: gate mode)")
    ap.add_argument("--control", action="store_true",
                    help="add the degenerate constant-depth control row")
    ap.add_argument("--holdout", action="append", default=None,
                    help="path substring of the val split (repeatable); "
                         "evaluates ONLY matching sequences")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"gmc": True, "tau_on": 0.1, "tau_off": 0.05} if args.gmc else {}
    if args.spatial_cache:
        kw["spatial_cache"] = True
    kw["gate_mode"] = args.gate_mode
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

    # pooled = pixel-weighted over the whole set; clipAbsRel/std = per-clip
    hdr = ("tau_on   active%  AbsRel   RMSE     d1      t-delta  OPW     "
           "TCE     clipAbsRel(std)  n")
    lines += [hdr, "-" * len(hdr)]
    per_clip = {}

    def row(label, m):
        per_clip[label] = m.pop("per_clip")
        lines.append(f"{label:<8} {m['active_ratio']*100:6.1f}  "
                     f"{m['absrel']:.4f}  {m['rmse']:7.4f}  {m['delta1']:.4f}  "
                     f"{m['temporal_delta']:.4f}   {m['opw']:.4f}  "
                     f"{m['tce']:.4f}  {m['absrel_clip']:.4f} "
                     f"({m['absrel_std']:.3f})  {m['n']}"
                     + (f" (+{m['skipped']} no-GT)" if m["skipped"] else ""))

    for tau_on, tau_off in taus:
        model.detector.tau_on, model.detector.tau_off = tau_on, tau_off
        row(f"{tau_on:g}", eval_once(model, loader, dev, args.max_clips))
    if args.control:
        # degenerate constant-depth control: proves t-delta/OPW alone can be
        # gamed, and that TCE cannot (REPORT.md §4.6 collapse scored 0.0000)
        row("const", eval_once(model, loader, dev, args.max_clips, constant=True))

    report = "\n".join(lines)
    print(report)
    with open(out, "a") as f:
        f.write(report + "\n\n")
    # per-clip values persisted: a different statistic (median, CI, per-dataset
    # split) never needs the model re-run
    scores = out.with_name(f"scores_{args.scores_tag or args.gate_mode}.json")
    scores.write_text(json.dumps(per_clip))
    print(f"\nappended -> {out}\nper-clip -> {scores}")


if __name__ == "__main__":
    main()
