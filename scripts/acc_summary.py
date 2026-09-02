"""One table out of every eval_acc.py dump. Reads JSON, runs nothing.

work_dirs/acc/table.txt is an append log -- it grows, it can hold two runs of
the same arm, and it is ordered by when things finished rather than by what
they mean. This rebuilds the comparison from the dumps themselves, so the
table always reflects the dumps on disk and a re-run simply overwrites its row.

    python scripts/acc_summary.py                       # every dump
    python scripts/acc_summary.py --gauge scaleshift/frame
    python scripts/acc_summary.py --regions
"""
import argparse
import json
import statistics
from pathlib import Path

OUT = Path("work_dirs/acc")
REGIONS = ("all", "edge", "flat", "dynamic", "static", "near", "mid", "far")


def summarise(path, gauge):
    d = json.loads(path.read_text())
    if "pooled" not in d or gauge not in d["pooled"]:
        return None
    pl, pc, fa = d["pooled"][gauge], d["per_clip"][gauge], d["fails"][gauge]
    srcs = list(pl)
    n = len(srcs)
    return {
        "tag": path.stem, "params": d["params_m"], "mode": d.get("mode", ""),
        "clips": sum(len(pc[s]["absrel"]) for s in srcs), "active": d["active"],
        "clip_len": json.loads(Path(d["manifest"]).read_text())["clip_len"],
        "absrel": sum(pl[s]["absrel"] for s in srcs) / n,
        "delta1": sum(pl[s]["delta1"] for s in srcs) / n,
        "med": sum(statistics.median(pc[s]["absrel"]) for s in srcs) / n,
        "p95": sum(sorted(pc[s]["absrel"])[int(0.95 * (len(pc[s]["absrel"]) - 1))]
                   for s in srcs) / n,
        "fail": sum(v["align"] for v in fa.values()),
        "cat": sum(v["catastrophic"] for v in fa.values()),
        "per_src": {s: pl[s]["absrel"] for s in srcs},
        "regions": (d.get("regions") or {}).get(gauge),
    }


# (label, key, lower_is_better) -- the axes a model can be ranked on. They
# disagree, which is the point: a model can lead on the typical clip and trail
# badly on the tail, and averaging the ranks would hide exactly that.
AXES = [
    ("MEAN AbsRel", "absrel", True),
    ("clip med", "med", True),
    ("P95 tail", "p95", True),
    ("delta1", "delta1", False),
    ("TUM", "tum", True),
    ("Bonn", "bonn", True),
    ("align fail", "fail", True),
    ("params(M)", "params", True),
]


def rank_table(rows, srcs):
    """Rank per axis, plus a Borda sum. The sum is a reading aid, not a
    verdict: it weights every axis equally, and `params` is a cost rather than
    a quality, so a model can win it by being small and mediocre."""
    for r in rows:
        for s in srcs:
            r[s] = r["per_src"].get(s, float("nan"))
    order = {}
    for label, key, lo in AXES:
        order[label] = sorted(rows, key=lambda r: (r[key] if lo else -r[key]))
    borda = {r["tag"]: sum(order[l].index(r) for l, _, _ in AXES) for r in rows}

    w = max(len(r["tag"]) for r in rows) + 9
    print(f"{'#':<3}" + "".join(f"{l:<{w}}" for l, _, _ in AXES))
    for i in range(len(rows)):
        cells = []
        for label, key, lo in AXES:
            r = order[label][i]
            v = r[key]
            cells.append(f"{r['tag']} {v:.0f}" if key == "fail" else
                         f"{r['tag']} {v:.1f}" if key == "params" else
                         f"{r['tag']} {v:.4f}")
        print(f"{i+1:<3}" + "".join(f"{c:<{w}}" for c in cells))
    print(f"\nBorda (rank 합, 낮을수록 상위; params 포함이라 작은 모델에 유리)")
    for i, (tag, b) in enumerate(sorted(borda.items(), key=lambda kv: kv[1])):
        print(f"  {i+1:<3}{tag:<{w}}{b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauge", default="scaleshift")
    ap.add_argument("--dir", default=str(OUT))
    ap.add_argument("--regions", action="store_true",
                    help="print the A6 breakdown for dumps that carry one")
    ap.add_argument("--rank", action="store_true",
                    help="rank the selected rows on each axis separately")
    ap.add_argument("-L", "--clip-len", type=int, default=None,
                    help="keep only dumps at this clip length")
    ap.add_argument("--only", action="append", default=None,
                    help="substring a tag must contain (repeatable, OR)")
    args = ap.parse_args()

    rows = [r for p in sorted(Path(args.dir).glob("*.json"))
            if (r := summarise(p, args.gauge))]
    if args.clip_len:
        rows = [r for r in rows if r["clip_len"] == args.clip_len]
    if args.only:
        rows = [r for r in rows if any(o in r["tag"] for o in args.only)]
    if not rows:
        print(f"no dump carries the {args.gauge!r} gauge in {args.dir}")
        return
    srcs = list(rows[0]["per_src"])
    if args.rank:
        print(f"gauge={args.gauge}  L={args.clip_len or 'all'}  "
              f"{len(rows)} rows\n")
        return rank_table(rows, srcs)
    print(f"gauge={args.gauge}\n")
    print(f"{'tag':<36}{'par':>7}{'L':>5}{'act%':>6}"
          + "".join(f"{s:>9}" for s in srcs)
          + f"{'MEAN':>8}{'med':>8}{'P95':>8}{'d1':>8}{'fail':>6}{'cat':>5}"
            f"{'n':>5}")
    for r in sorted(rows, key=lambda r: (r["clip_len"], r["tag"])):
        print(f"{r['tag']:<36}{r['params']:7.1f}{r['clip_len']:5d}"
              f"{r['active']*100:6.0f}"
              + "".join(f"{r['per_src'].get(s, float('nan')):9.4f}" for s in srcs)
              + f"{r['absrel']:8.4f}{r['med']:8.4f}{r['p95']:8.4f}"
                f"{r['delta1']:8.4f}{r['fail']:6d}{r['cat']:5d}{r['clips']:5d}")

    if args.regions:
        for r in sorted(rows, key=lambda r: (r["clip_len"], r["tag"])):
            if not r["regions"]:
                continue
            print(f"\n{r['tag']} regions (AbsRel / delta1 / % of valid px)")
            for name in REGIONS:
                cells = []
                for s in srcs:
                    t = (r["regions"].get(s) or {}).get(name)
                    tot = r["regions"][s]["all"]["px"] or 1.0
                    cells.append(f"{'--':>22}" if not t or not t["px"] else
                                 f"{t['rel']/t['px']:10.4f}{t['d1']/t['px']:7.3f}"
                                 f"{t['px']/tot*100:5.0f}")
                print(f"  {name:<9}" + "".join(cells))


if __name__ == "__main__":
    main()
