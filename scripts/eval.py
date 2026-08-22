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

from sokkanaem import checkpoint_config, from_checkpoint
from sokkanaem.data import build_mixed, even_subset
from sokkanaem.metrics import clip_scores, pooled


@torch.no_grad()
def eval_once(model, loader, dev, max_clips, constant=False, align="median"):
    """constant=True replaces the prediction with the clip's best possible
    constant depth — the degenerate control. It scores t-delta 0 and OPW 0
    while being useless, which is exactly why TCE is reported (metrics.py)."""
    acc = {k: [] for k in ("absrel", "rmse", "delta1", "temporal_delta",
                           "opw", "tce", "active_ratio", "absrel_metric",
                           "scale", "scale_drift", "_pooled")}
    skipped = 0
    for ci, (clip, gt, valid) in enumerate(loader):
        if ci >= max_clips:
            break
        clip, gt, valid = clip.to(dev), gt.to(dev), valid.to(dev)
        depths, masks = model.forward_clip(clip)  # detector-driven masks
        v = valid.bool()
        if constant:
            depths = torch.full_like(gt, gt[v].median())
        if align == "scaleshift":
            # Least-squares scale+shift in disparity space -- the MiDaS
            # protocol relative-depth baselines are designed for. It has two
            # degrees of freedom against median scaling's one, so it flatters
            # any model it is applied to; the only reason to run it here is to
            # compare against those baselines on THEIR alignment rather than
            # letting the protocol carry the result.
            import numpy as np
            dg = (1.0 / gt.clamp(min=1e-3))[v].double().cpu().numpy()
            dp = (1.0 / depths.clamp(min=1e-3))[v].double().cpu().numpy()
            A = np.stack([dp, np.ones_like(dp)], axis=1)
            (a, b), *_ = np.linalg.lstsq(A, dg, rcond=None)
            aligned = a * (1.0 / depths.clamp(min=1e-3)) + b
            p = 1.0 / aligned.clamp(min=1e-3)
            # the scale-drift diagnostic below is defined for the 1-DOF fit;
            # under a 2-DOF fit report the median ratio so the column stays
            # populated and comparable in magnitude, not identical in meaning
            s = gt[v].median() / depths[v].median().clamp(min=1e-6)
        else:
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
        # Median scaling scores relative structure only. A fixed camera needs
        # metres, so report the UNSCALED error too, plus how far the per-clip
        # scale sits from 1 and how much it moves inside the clip (short-
        # horizon scale drift — the thing a long stream accumulates).
        acc["absrel_metric"].append(
            ((depths[v] - gt[v]).abs() / gt[v].clamp(min=1e-6)).mean().item())
        acc["scale"].append(s.item())
        fs = [(gt[0, t][vt].median()
               / depths[0, t][vt].median().clamp(min=1e-6)).item()
              for t in range(gt.shape[1]) if (vt := v[0, t]).any()]
        fs = torch.tensor(fs)
        acc["scale_drift"].append(
            (fs.std() / fs.mean().clamp(min=1e-6)).item() if len(fs) > 1 else 0.0)
    # pixel-pooled dataset-level metrics are the primary numbers (convention,
    # and not hostage to a few blown-up clips); per-clip std reports spread
    sums = acc.pop("_pooled")
    out = dict(pooled(sums))
    t = {k: torch.tensor(v) for k, v in acc.items()}
    for k in ("active_ratio", "absrel_metric", "scale", "scale_drift"):
        out[k] = t[k].mean().item()
    out["n"] = len(t["absrel"])
    out["skipped"] = skipped
    out["absrel_std"] = t["absrel"].std().item() if out["n"] > 1 else 0.0
    out["absrel_clip"] = t["absrel"].mean().item()
    out["per_clip"] = {k: v.tolist() for k, v in t.items()}
    out["sums"] = sums  # kept so several sources can be re-pooled by pixel
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", action="append", required=True,
                    help="dataset spec 'name:/root[:scale]', repeatable")
    ap.add_argument("--size", type=int, default=None,
                    help="eval resolution; default = the checkpoint's "
                         "training size (128 if unrecorded)")
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--clip-stride", type=int, default=None,
                    help="clip start spacing; defaults to --clip-len (disjoint "
                         "clips). A shorter stride buys clips out of a fixed "
                         "holdout at the cost of their independence, which is "
                         "the only way a 256-frame protocol gets a sample.")
    ap.add_argument("--max-clips", type=int, default=100,
                    help="clips PER SOURCE. Reading one concatenated stream "
                         "instead made 'synthetic 500' mean 'VKITTI2 500' "
                         "(reports/20260729.md §7.1)")
    ap.add_argument("--sweep-tau", action="store_true",
                    help="sweep detector threshold: skip-vs-accuracy curve")
    ap.add_argument("--gmc", action="store_true",
                    help="ego-motion mode: Low-Res GMC + feature gating (§3.5)")
    ap.add_argument("--spatial-cache", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="reuse static-patch spatial outputs (§4.5 wall-clock); "
                         "default = whatever the checkpoint was trained with")
    ap.add_argument("--temporal-cache", action=argparse.BooleanOptionalAction,
                    default=None, help="reuse static-patch temporal readout")
    ap.add_argument("--gate-mode", default="delta", choices=["delta", "drop"],
                    help="§4.4 gating-position ablation: Δ-gating vs token drop")
    ap.add_argument("--scores-tag", default=None,
                    help="name for the per-clip JSON dump (default: gate mode)")
    ap.add_argument("--bin-temp", type=float, default=None,
                    help="sharpen the binned head's softmax at inference "
                         "(Section 6.5's range-compression test)")
    ap.add_argument("--dense-above", type=float, default=None,
                    help="route frames above this activity through the dense "
                         "path (Section 6.3). It is an accuracy/compute knob "
                         "with a large effect, so it is swept, not assumed.")
    ap.add_argument("--refresh", default=None,
                    choices=["keyframe", "rolling"],
                    help="rolling refreshes 1/K of the patches every frame "
                         "instead of all of them every K frames: same "
                         "amortised cost, no refresh discontinuity")
    ap.add_argument("--keyframe-every", type=int, default=None,
                    help="override the checkpoint's keyframe refresh period. "
                         "REPORT 4.30 shows accuracy sawtooths between "
                         "keyframes, so this is an accuracy/compute knob, not "
                         "just a safety valve")
    ap.add_argument("--align", default="median",
                    choices=["median", "scaleshift"],
                    help="per-clip alignment. median (default) is 1-DOF and "
                         "what every SOKKANAEM number uses; scaleshift is the "
                         "2-DOF disparity-space fit relative-depth baselines "
                         "are evaluated under, provided so the two can be "
                         "compared without the protocol carrying the result")
    ap.add_argument("--control", action="store_true",
                    help="add the degenerate constant-depth control row")
    ap.add_argument("--holdout", action="append", default=None,
                    help="path substring of the val split (repeatable); "
                         "evaluates ONLY matching sequences")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = checkpoint_config(args.ckpt)
    if args.size is None:
        args.size = cfg.get("size", 128)
    kw = {"gmc": True, "tau_on": 0.1, "tau_off": 0.05} if args.gmc else {}
    kw.update({k: v for k, v in (("spatial_cache", args.spatial_cache),
                                 ("temporal_cache", args.temporal_cache))
               if v is not None})
    kw["gate_mode"] = args.gate_mode
    if args.keyframe_every is not None:
        kw["keyframe_every"] = args.keyframe_every
    if args.refresh is not None:
        kw["refresh"] = args.refresh
    # trained [model] kwargs come from config.toml next to the ckpt
    model = from_checkpoint(args.ckpt, dev, **kw).eval()
    if args.dense_above is not None:
        model.dense_above = args.dense_above
    if args.bin_temp is not None:
        model.decoder.bin_temp = args.bin_temp

    dataset, _ = build_mixed(args.data, clip_len=args.clip_len,
                             clip_stride=args.clip_stride or args.clip_len, size=args.size,
                             holdout=args.holdout, val=True)
    # one loader per source: build_mixed concatenates in spec order, and a
    # single stream truncated at --max-clips only ever reached the first one
    sources = [(spec.split(":")[0],
                torch.utils.data.DataLoader(even_subset(ds, args.max_clips),
                                            batch_size=1, shuffle=False))
               for spec, ds in zip(args.data, dataset.datasets)]

    out = Path(args.ckpt).parent / "eval.txt"
    # every knob that changes the numbers belongs in the header: a row whose
    # keyframe period / clip length / tag is unrecorded cannot be traced back
    # to its run, and a paper table then gets assembled from two of them
    lines = [f"ckpt={args.ckpt} data={args.data} clips={len(dataset)} "
             f"size={args.size} max={args.max_clips}/source "
             f"clip_len={args.clip_len} stride={args.clip_stride or args.clip_len} align={args.align} "
             f"gate_mode={args.gate_mode} "
             f"keyframe_every={args.keyframe_every or model.detector.keyframe_every} "
             f"refresh={model.detector.refresh} "
             f"dense_above={model.dense_above} bin_temp={getattr(model.decoder, 'bin_temp', 1.0)} gmc={args.gmc} tag={args.scores_tag or args.gate_mode} "
             f"spatial_cache={model.spatial_cache} "
             f"temporal_cache={model.temporal_cache}"]

    if not args.sweep_tau:
        taus = [(model.detector.tau_on, model.detector.tau_off)]
    elif args.gmc:  # relative-L1 feature scale (§3.5)
        taus = [(0.0, 0.0), (0.05, 0.025), (0.1, 0.05), (0.2, 0.1),
                (0.4, 0.2), (0.8, 0.4)]
    else:
        taus = [(0.0, 0.0), (0.005, 0.0025), (0.01, 0.005), (0.02, 0.01),
                (0.05, 0.025), (0.1, 0.05)]

    # pooled = pixel-weighted; clipAbsRel/std = per-clip. mAbsRel/scale/drift
    # are the UNSCALED numbers: median scaling hides absolute-scale error.
    hdr = ("tau_on   source        active%  AbsRel   RMSE     d1      t-delta  "
           "OPW     TCE     mAbsRel  scale  drift   clipAbsRel(std)  n")
    lines += [hdr, "-" * len(hdr)]
    per_clip = {}

    def row(label, src, m):
        pc = m.pop("per_clip", None)
        if pc is not None:
            per_clip[f"{label}/{src}"] = pc
        lines.append(f"{label:<8} {src:<13} {m['active_ratio']*100:6.1f}  "
                     f"{m['absrel']:.4f}  {m['rmse']:7.4f}  {m['delta1']:.4f}  "
                     f"{m['temporal_delta']:.4f}   {m['opw']:.4f}  "
                     f"{m['tce']:.4f}  {m['absrel_metric']:.4f}  "
                     f"{m['scale']:.3f}  {m['scale_drift']:.4f}  "
                     f"{m['absrel_clip']:.4f} "
                     f"({m['absrel_std']:.3f})  {m['n']}"
                     + (f" (+{m['skipped']} no-GT)" if m["skipped"] else ""))

    CLIPK = ("active_ratio", "absrel_metric", "scale", "scale_drift",
             "absrel_clip", "absrel_std")

    def combine(ms, by_pixel):
        """by_pixel=True: one number over every pixel of every source, so the
        biggest dataset dominates. False: equal weight per source. Report both
        — they are different claims and the old single row conflated them."""
        n = [m["n"] for m in ms]
        w = ([x / max(sum(n), 1) for x in n] if by_pixel
             else [1 / len(ms)] * len(ms))
        keys = ("absrel", "rmse", "delta1", "temporal_delta", "opw", "tce")
        c = (dict(pooled([s for m in ms for s in m["sums"]])) if by_pixel
             else {k: sum(wi * m[k] for wi, m in zip(w, ms)) for k in keys})
        c.update({k: sum(wi * m[k] for wi, m in zip(w, ms)) for k in CLIPK})
        c["n"], c["skipped"] = sum(n), sum(m["skipped"] for m in ms)
        return c

    def run(label, **kw):
        ms = []
        kw.setdefault("align", args.align)
        for src, loader in sources:
            m = eval_once(model, loader, dev, args.max_clips, **kw)
            row(label, src, m)
            ms.append(m)
        if len(ms) > 1:
            row(label, "MEAN(src)", combine(ms, by_pixel=False))
            row(label, "POOLED(px)", combine(ms, by_pixel=True))

    for tau_on, tau_off in taus:
        model.detector.tau_on, model.detector.tau_off = tau_on, tau_off
        run(f"{tau_on:g}")
    if args.control:
        # degenerate constant-depth control: proves t-delta/OPW alone can be
        # gamed, and that TCE cannot (REPORT.md §4.6 collapse scored 0.0000)
        run("const", constant=True)

    report = "\n".join(lines)
    print(report)
    with open(out, "a") as f:
        f.write(report + "\n\n")
    # per-clip values persisted: a different statistic (median, CI, per-dataset
    # split) never needs the model re-run
    gmc = getattr(model, "gmc", None)
    if gmc is not None and getattr(gmc, "calls", 0):
        lines.append(f"gmc: {gmc.fallbacks}/{gmc.calls} frames fell back to "
                     f"identity ({100 * gmc.fallbacks / gmc.calls:.1f}%)")
        print(lines[-1])

    scores = out.with_name(f"scores_{args.scores_tag or args.gate_mode}.json")
    scores.write_text(json.dumps(per_clip))
    print(f"\nappended -> {out}\nper-clip -> {scores}")


if __name__ == "__main__":
    main()
