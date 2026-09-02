"""Clip-level bootstrap CI for the comparison tables (r2 Major 2).

Table 3a is 13 disjoint clips, so a few-percent gap is not resolvable and the
draft says so. This puts a number on it instead of a disclaimer: resample the
clips (stratified by source, since the table is a dataset-balanced mean) and
report the percentile interval, plus the PAIRED interval on the baseline/ours
ratio -- paired because every model saw the same clips, and the between-clip
variance is far larger than the between-model gap.

No model is re-run: eval.py and metrics.report already persist per-clip values.

    python scripts/bootstrap_ci.py --ours work_dirs/v11-longclip-spread-s0/scores_L256K30.json
    python scripts/bootstrap_ci.py --ours .../scores_L8K30.json --suffix l8 --metric absrel

Caveat on AbsRel: the draft's rows are pixel-pooled within a source and then
balanced, while a per-clip dump can only be averaged per clip, so the AbsRel
centre here sits a little off the table (0.1962 vs 0.1907 at L256). t-delta,
OPW and TCE are per-pixel-uniform across clips, so their centres match the
table exactly. The interval WIDTH is what this script is for.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

SRC = ("tum", "bonn")
METRICS = ("absrel", "delta1", "temporal_delta", "opw", "tce",
           "scale_drift", "scale_logstd", "scale_step")
BASE = Path("work_dirs/baselines")


def ours(path, label="0.05"):
    d = json.load(open(path))
    return {s: d[f"{label}/{s}"] for s in SRC}


def baselines(suffix):
    """{model: {source: {metric: [per-clip]}}} for every baseline dump that
    exists for ALL sources at this suffix -- a model measured on one source
    only cannot enter a dataset-balanced comparison."""
    out = {}
    for p in sorted(BASE.glob(f"*-{SRC[0]}-{suffix}.json")):
        name = re.sub(f"-{SRC[0]}-{suffix}$", "", p.stem)
        paths = {s: BASE / f"{name}-{s}-{suffix}.json" for s in SRC}
        if all(q.exists() for q in paths.values()):
            out[name] = {s: json.load(open(q)) for s, q in paths.items()}
    return out


def draws(per_src, metric, idx):
    """Dataset-balanced mean of one metric under the resampled clip indices."""
    return np.mean([np.asarray(per_src[s][metric])[idx[s]].mean(axis=-1)
                    for s in SRC], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="scores_*.json from eval.py")
    ap.add_argument("--label", default="0.05", help="tau label inside --ours")
    ap.add_argument("--suffix", default="l256", help="baseline dump suffix")
    ap.add_argument("--metric", action="append", choices=METRICS,
                    help="repeatable; default all")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    us, bl = ours(args.ours, args.label), baselines(args.suffix)
    n = {s: len(us[s]["absrel"]) for s in SRC}
    for name, b in bl.items():
        bad = [s for s in SRC if len(b[s]["absrel"]) != n[s]]
        assert not bad, f"{name}: {bad} clip count differs from ours -- the " \
                        "paired bootstrap needs the same clips"

    rng = np.random.default_rng(args.seed)
    # ONE index draw shared by every model: that is what makes it paired
    idx = {s: rng.integers(0, n[s], size=(args.reps, n[s])) for s in SRC}
    ident = {s: np.arange(n[s])[None] for s in SRC}
    lo, hi = 2.5, 97.5

    print(f"ours={args.ours} label={args.label} suffix={args.suffix} "
          f"clips={ {s: n[s] for s in SRC} } reps={args.reps} seed={args.seed}")
    for metric in (args.metric or METRICS):
        u, ud = draws(us, metric, ident)[0], draws(us, metric, idx)
        print(f"\n{metric}: ours {u:.4f}  95% CI "
              f"[{np.percentile(ud, lo):.4f}, {np.percentile(ud, hi):.4f}]")
        print(f"  {'baseline':<44} {'point':>7} {'95% CI':>19} "
              f"{'ratio':>7} {'ratio 95% CI':>17}  P(ours better)")
        for name, b in sorted(bl.items()):
            if metric not in b[SRC[0]]:
                # the scale statistics postdate these dumps -- a baseline
                # measured before them re-enters the table on its next run
                print(f"  {name:<44}       -- not in dump (re-run needed)")
                continue
            v, vd = draws(b, metric, ident)[0], draws(b, metric, idx)
            # delta1 is higher-is-better; every other metric here is lower
            r, rd = ((u / v, ud / vd) if metric == "delta1" else (v / u, vd / ud))
            win = (ud > vd) if metric == "delta1" else (ud < vd)
            print(f"  {name:<44} {v:7.4f} "
                  f"[{np.percentile(vd, lo):7.4f},{np.percentile(vd, hi):7.4f}] "
                  f"{r:7.3f} [{np.percentile(rd, lo):6.3f},"
                  f"{np.percentile(rd, hi):6.3f}]  {win.mean():.3f}")


if __name__ == "__main__":
    main()
