"""Generate the comparison-table rows from the measurement logs.

Table 11 of the draft was wrong because a row was assembled by hand from two
runs. Rows that are generated cannot be assembled by hand.

Reads the baseline logs (one "pooled :" line per model per source) and our own
eval.txt blocks (one MEAN(src) row per run), groups by model and tag, averages
over the sources of a domain, and prints markdown.

    python scripts/make_tables.py work_dirs/r1-round.log
    python scripts/make_tables.py work_dirs/r1-round.log --tag L256
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL = ("tum", "bonn")
SYNTH = ("vkitti2", "tartanair2", "pointodyssey")
COLS = ("absrel", "delta1", "t-delta", "opw", "tce")
# "DA-v1-Small (24.8M) tum L8 on 89 holdout clips ..." / "DA3-BASE (0.12B) [scaleshift] bonn L8 on ..."
# "DA-v1-Small (24.8M) tum L8 on 89 holdout clips", and the VDA script's
# "Video-Depth-Anything-Small metric (~28.4M) tum L8 on ..."
HEAD = re.compile(r"^(?P<model>[\w.\-]+(?: metric)?(?: \[\w+\])?) "
                  r"\(~?(?P<params>[\d.]+[MB])\)"
                  r"(?P<rest>[^:]*) on (?P<n>\d+) holdout clips")
POOLED = re.compile(r"pooled\s*:\s*(.*)")


def baseline_rows(log):
    """{(model, tag): {source: {metric: value}}} from a baseline-script log."""
    runs, key = {}, None
    for line in Path(log).read_text(errors="replace").splitlines():
        m = HEAD.match(line.strip())
        if m:
            words = m.group("rest").split()
            src = next((w for w in words if w in REAL + SYNTH), "?")
            tag = " ".join(w for w in words if w not in REAL + SYNTH) or "-"
            key = (f"{m.group('model')} ({m.group('params')})", tag, src,
                   int(m.group("n")))
            continue
        p = POOLED.search(line)
        if p and key:
            vals = dict(re.findall(r"([\w\-]+)=([\d.]+)", p.group(1)))
            model, tag, src, n = key
            runs.setdefault((model, tag), {})[src] = {
                k.lower(): float(v) for k, v in vals.items()} | {"n": n}
            key = None
    return runs


def ours_rows(paths):
    """{(ckpt, tag): {'MEAN': {...}}} from eval.txt blocks."""
    out = {}
    for path in paths:
        if not Path(path).exists():
            continue
        head = None
        for line in Path(path).read_text(errors="replace").splitlines():
            if line.startswith("ckpt="):
                head = dict(re.findall(r"(\w+)=(\S+)", line))
                continue
            if "MEAN(src)" in line and head:
                f = line.split()
                if not f[0].replace(".", "").isdigit():
                    continue  # 'const' row: the constant-depth control
                out[(head.get("ckpt", "?"), head.get("tag", "-"))] = {
                    "active": float(f[2]), "absrel": float(f[3]),
                    "delta1": float(f[5]), "t-delta": float(f[6]),
                    "opw": float(f[7]), "tce": float(f[8]), "n": int(f[-1]),
                    "clip_len": head.get("clip_len", "?"),
                    "keyframe": head.get("keyframe_every", "?"),
                }
    return out


def mean_over(per_src, sources):
    got = [v for s, v in per_src.items() if s in sources]
    if not got:
        return None
    return {k: sum(g[k] for g in got) / len(got)
            for k in got[0] if k != "n"} | {"n": sum(g["n"] for g in got),
                                            "sources": len(got)}


def seed_table(paths, cols=COLS):
    """mean +- std over seeds, per (tag, clip_len, keyframe).

    Seed variance was only ever characterised at 8k steps on a different
    recipe, so the final configuration carries its own spread: same tags, one
    eval.txt per seed, aggregated here rather than by hand.
    """
    import statistics as st
    rows = {}
    for path in paths:
        for (ck, tag), m in ours_rows([path]).items():
            key = (tag, m["clip_len"], m["keyframe"])
            rows.setdefault(key, []).append(m)
    print("\n*final configuration, mean +- std over seeds*\n")
    print("| tag | clip | K | active% | " + " | ".join(cols) + " | seeds |")
    print("|---|---:|---:|---:|" + "---:|" * (len(cols) + 1))
    for (tag, cl, kf), ms in sorted(rows.items()):
        cells = []
        for c in cols:
            vals = [m[c] for m in ms]
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{st.fmean(vals):.4f} ± {sd:.4f}")
        act = st.fmean([m["active"] for m in ms])
        print(f"| {tag} | {cl} | {kf} | {act:.1f} | " + " | ".join(cells)
              + f" | {len(ms)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", default=[])
    ap.add_argument("--tag", default=None, help="only rows with this tag")
    ap.add_argument("--seeds", nargs="*", default=None,
                    help="eval.txt paths of seed replicates: prints mean +- std "
                         "per tag instead of one row per checkpoint")
    ap.add_argument("--ours", nargs="*", default=None,
                    help="eval.txt paths (default: every work_dirs/*/eval.txt)")
    args = ap.parse_args()

    if args.seeds:
        seed_table(args.seeds)
        return

    runs = {}
    for log in args.logs:
        runs.update(baseline_rows(log))

    for dom, sources in (("real", REAL), ("synthetic", SYNTH)):
        rows = []
        for (model, tag), per_src in sorted(runs.items()):
            if args.tag and args.tag not in tag:
                continue
            m = mean_over(per_src, sources)
            if m is None:
                continue
            rows.append((m["absrel"], model, tag, m))
        if not rows:
            continue
        print(f"\n*{dom} holdout, mean over {len(sources)} sources*\n")
        print("| Model | tag | " + " | ".join(COLS) + " | srcs | clips |")
        print("|---|---|" + "---:|" * (len(COLS) + 2))
        for _, model, tag, m in sorted(rows):
            print(f"| {model} | {tag} | "
                  + " | ".join(f"{m[c]:.4f}" if c in m else "-" for c in COLS)
                  + f" | {m['sources']} | {m['n']} |")

    paths = args.ours if args.ours is not None else sorted(
        str(p) for p in ROOT.glob("work_dirs/*/eval.txt"))
    ours = ours_rows(paths)
    if ours:
        print("\n*ours, MEAN(src) rows from eval.txt*\n")
        print("| ckpt | tag | clip | K | active% | " + " | ".join(COLS) + " | clips |")
        print("|---|---|---:|---:|---:|" + "---:|" * (len(COLS) + 1))
        for (ck, tag), m in sorted(ours.items()):
            if args.tag and args.tag not in tag:
                continue
            print(f"| {Path(ck).parent.name} | {tag} | {m['clip_len']} | "
                  f"{m['keyframe']} | {m['active']:.1f} | "
                  + " | ".join(f"{m[c]:.4f}" for c in COLS)
                  + f" | {m['n']} |")


if __name__ == "__main__":
    main()
