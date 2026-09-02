"""Check a model against PLAN_ACC's gates. Reads dumps, runs nothing.

The gates exist so that "better than before" stops being the standard, and a
gate nobody can run is a gate nobody applies. Every threshold here is quoted
from PLAN_ACC §1.3 and each line prints the measured value next to it, so a
near-miss is visible rather than hidden behind a PASS/FAIL.

    python scripts/acc_gate.py work_dirs/acc/ours-L8.json \
        work_dirs/acc/ours-L32.json work_dirs/acc/ours-L256.json

The first dump is the G1 subject (it must be an L8 run); any others feed G2,
matched by their manifest's clip length. Criteria needing several seeds or
per-frame traces are printed as SKIP with what they need -- silence would read
as a pass.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

GAUGE = "scaleshift"   # G1's primary gauge: the only one fair to both families
# G2's growth criteria are written against the clip-wide fit, and measured
# there a STATELESS DPT-Large loses 116% from 8 to 256 frames -- the number
# is dominated by one gauge failing to cover a longer stream, not by drift.
# --gauge scaleshift/frame reads the per-frame dumps, where what is left is
# the model's own depth-shape drift. Report both; neither alone is the answer.

# PLAN.md §3.1 -- the CURRENT promotion gate, and the one that decides whether
# a checkpoint is reported. G1 below is PLAN_ACC's older, looser set; both are
# printed because a run between the two is exactly the interesting case.
# The first four lines are DPT-Large's measured level (no alignment failures);
# the two Bonn lines are DA2-small's real per-source performance, so that one
# source's average cannot hide the other's collapse.
P1 = [
    ("balanced AbsRel", "absrel", "<=", 0.090),
    ("balanced delta1", "delta1", ">=", 0.930),
    ("clip P95 AbsRel", "p95_absrel", "<=", 0.153),
    ("alignment failures", "align_fail", "<=", 0),
    ("catastrophic clips (AbsRel>1)", "catastrophic", "<=", 0),
    ("TUM clip median AbsRel", "tum_med_absrel", "<=", 0.095),
    ("Bonn pooled AbsRel", "bonn_absrel", "<=", 0.058),
    ("Bonn delta1", "bonn_delta1", ">=", 0.973),
]

# PLAN.md §3.2 -- sharpness. (label, key, comparison, reference) where the
# reference is another row of the same sharpness JSONL: a gradient ratio can be
# bought with noise, so the gate is jointly on F1, flat TV and overshoot.
P2 = [
    ("balanced grad_ratio", "grad_ratio", ">=", 0.756),
    ("boundary F1 vs DA2", "boundary_f1", ">=", "da2"),
    ("edge AbsRel vs DA2", "absrel_edge", "<=", "da2"),
    ("overshoot vs DA2", "overshoot", "<=", "da2"),
    ("flat TV vs reported", "flat_tv", "<=", "reported"),
]

# PLAN.md §3.3 -- long-stream gate, on the PER-FRAME gauge (--gauge
# scaleshift/frame). The clip-wide fit is dominated by one 2-DOF window having
# to cover 256 frames, which is why a stateless DPT-Large "drifts" 116% there
# and 0% here (REPORT §4.43); the per-frame numbers are the model's own drift.
P3 = [
    ("L8->L256 AbsRel growth", 0.05),      # <= +5%
    ("L8->L256 delta1 drop (pt)", 1.0),    # <= 1.0pt
]

# (label, key, comparison, threshold) -- see PLAN_ACC §1.3
G1 = [
    ("MEAN(src) AbsRel", "absrel", "<=", 0.100),
    ("clip medAbsRel", "med_absrel", "<=", 0.090),
    ("MEAN(src) delta1", "delta1", ">=", 0.910),
    ("TUM AbsRel", "tum_absrel", "<=", 0.115),
    ("Bonn AbsRel", "bonn_absrel", "<=", 0.070),
    ("catastrophic clips (AbsRel>1)", "catastrophic", "<=", 0),
]


def read(path, gauge=GAUGE):
    d = json.loads(Path(path).read_text())
    assert "pooled" in d, (
        f"{path}: dump predates the pooled-numbers field. Re-run "
        f"scripts/eval_acc.py for it -- deriving the gate's numbers from the "
        f"per-clip lists instead would weight them differently than the table "
        f"they are checking, which is the confusion PLAN_ACC A2 exists to end.")
    assert gauge in d["pooled"], (
        f"{path}: no {gauge!r} gauge; dump has {list(d['pooled'])}")
    pl, pc = d["pooled"][gauge], d["per_clip"][gauge]
    srcs = list(pl)
    L = json.loads(Path(d["manifest"]).read_text())["clip_len"] \
        if Path(d["manifest"]).exists() else None
    return {
        "label": d["label"], "path": str(path), "clip_len": L,
        "absrel": sum(pl[s]["absrel"] for s in srcs) / len(srcs),
        "delta1": sum(pl[s]["delta1"] for s in srcs) / len(srcs),
        "med_absrel": sum(statistics.median(pc[s]["absrel"])
                          for s in srcs) / len(srcs),
        "catastrophic": sum(v["catastrophic"] for v in d["fails"][gauge].values()),
        "align_fail": sum(v["align"] for v in d["fails"][gauge].values()),
        # P95 per source then averaged, like every other balanced number here:
        # pooling the clips instead would let the source with more clips set
        # the tail
        "p95_absrel": sum(_p95(pc[s]["absrel"]) for s in srcs) / len(srcs),
        **{f"{s}_absrel": pl[s]["absrel"] for s in srcs},
        **{f"{s}_delta1": pl[s]["delta1"] for s in srcs},
        **{f"{s}_med_absrel": statistics.median(pc[s]["absrel"]) for s in srcs},
    }


def _p95(xs):
    v = sorted(xs)
    return v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))] if v else float("nan")


def sharp_rows(paths):
    """label -> balanced sharpness numbers, merged over every --sharp source.

    Two shapes are accepted, because the two protocols exist for different
    jobs: a scripts/sharp_metric.py JSONL (quick arm screening, its own 100
    clips per source) and an eval_acc.py --sharp dump (the SEALED manifest, the
    same clips P1 is measured on). Later files win, so pass the sealed dump
    last when both name the same arm."""
    rows = {}
    for path in paths:
        text = Path(path).read_text()
        if str(path).endswith(".jsonl"):
            for line in text.splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[r["label"]] = r["scores"]["balanced"]
        else:
            d = json.loads(text)
            assert d.get("sharp"), (
                f"{path}: no sharpness block -- re-run eval_acc.py with --sharp")
            srcs = list(d["sharp"])
            keys = {k for s in srcs for k in d["sharp"][s]}
            rows[d["label"]] = {k: sum(d["sharp"][s][k] for s in srcs) / len(srcs)
                                for k in keys}
    return rows


# the same arm is labelled by the screening script and by eval_acc, and the
# two spellings differ; either satisfies a reference
DA2 = ("Depth-Anything-V2-Small-hf", "depth-anything/Depth-Anything-V2-Small-hf")
REPORTED = ("ours (reported v11)", "v11-longclip-spread-s0")


def sharpness(paths, subject, da2=DA2, reported=REPORTED):
    """P2: the subject's row against the DA2 row and the reported checkpoint's.
    Returns (ok, missing) -- a reference absent from every file is reported,
    never silently treated as a pass."""
    rows = sharp_rows(paths)
    def pick(names):
        return next((n for n in names if n in rows), None)
    refs = {"da2": pick(da2), "reported": pick(reported)}
    missing = ([subject] if subject not in rows else []) + \
        [k for k, v in refs.items() if v is None]
    if missing:
        return False, missing
    ok = True
    for label, key, op, ref in P2:
        want = ref if isinstance(ref, float) else rows[refs[ref]][key]
        ok &= check(label, rows[subject][key], op, want)
    return ok, []


def check(label, got, op, want, fmt="{:.4f}"):
    ok = got <= want if op == "<=" else got >= want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<32} "
          f"{fmt.format(got)} {op} {fmt.format(want)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", help="eval_acc.py JSON dumps; the "
                                             "first must be the L8 run")
    ap.add_argument("--sharp", action="append", default=None,
                    help="repeatable: sharpness JSONL (sharp_metric.py --out) "
                         "or an eval_acc.py --sharp dump. P2 needs the DA2 and "
                         "reported rows too, so pass whichever files hold them")
    ap.add_argument("--sharp-label", default=None,
                    help="which label is the subject; defaults to the label of "
                         "the first accuracy dump")
    ap.add_argument("--gauge", default=GAUGE,
                    help="which alignment's numbers to gate on; "
                         "'scaleshift/frame' reads a --per-frame dump")
    args = ap.parse_args()

    runs = [read(p, args.gauge) for p in args.dumps]
    base = runs[0]
    by_len = {r["clip_len"]: r for r in runs}
    print(f"{base['label']}  gauge={args.gauge}  {base['path']}\n")

    print("P1 - PLAN.md §3.1 promotion gate (8-frame, real TUM+Bonn)")
    if base["clip_len"] != 8:
        print(f"  [SKIP] first dump is a {base['clip_len']}-frame run; "
              f"P1 is defined at 8")
        p1 = False
    else:
        p1 = all([check(lbl, base[k], op, thr,
                        "{:.0f}" if k in ("catastrophic", "align_fail")
                        else "{:.4f}")
                  for lbl, k, op, thr in P1])

    print("\nP2 - PLAN.md §3.2 sharpness gate")
    subject = args.sharp_label or base["label"]
    if not args.sharp:
        print("  [SKIP] pass --sharp <file> (an eval_acc.py --sharp dump, or "
              "a sharp_metric.py JSONL)")
        p2 = False
    else:
        p2, missing = sharpness(args.sharp, subject)
        if missing:
            print(f"  [SKIP] labels absent from {args.sharp}: {missing}")

    print("\nP3 - PLAN.md §3.3 long-stream gate")
    by_len_local = {r["clip_len"]: r for r in runs}
    if 8 not in by_len_local or 256 not in by_len_local:
        print("  [SKIP] needs an L8 and an L256 dump")
        p3 = False
    else:
        a8, a256 = by_len_local[8]["absrel"], by_len_local[256]["absrel"]
        d8, d256 = by_len_local[8]["delta1"], by_len_local[256]["delta1"]
        p3 = check(P3[0][0], (a256 / a8 - 1) * 100, "<=", P3[0][1] * 100,
                   "{:.1f}%")
        p3 &= check(P3[1][0], (d8 - d256) * 100, "<=", P3[1][1], "{:.2f}")
        if args.gauge != "scaleshift/frame":
            print(f"  [note] gauge is {args.gauge}; §3.3 is written on the "
                  f"per-frame gauge (--gauge scaleshift/frame)")
    print("  [SKIP] sparse-vs-dense L8 gap (<=2%) and sharpness drop after the "
          "sparse switch (<=3%): need the tau=0 dump and its sharpness block")

    print("\nG1 - relative depth accuracy (8-frame, real TUM+Bonn)")
    if base["clip_len"] != 8:
        print(f"  [SKIP] first dump is a {base['clip_len']}-frame run; "
              f"G1 is defined at 8")
        g1 = False
    else:
        g1 = all([check(lbl, base[k], op, thr,
                        "{:.0f}" if k == "catastrophic" else "{:.4f}")
                  for lbl, k, op, thr in G1])
    if base["align_fail"]:
        print(f"  [note] {base['align_fail']} clips needed a disparity fit "
              f"that crossed zero; their depth is clamped fiction")
    print("  [SKIP] 3-seed agreement and non-overlapping CI vs the current "
          "model: needs 3 seed dumps (run this per seed)")

    print("\nG2 - long-stream accuracy")
    g2 = True
    if 8 not in by_len:
        print("  [SKIP] no 8-frame run to compare against")
        g2 = False
    else:
        a8, d8 = by_len[8]["absrel"], by_len[8]["delta1"]
        for L, lim in ((32, 1.05), (256, 1.10)):
            if L not in by_len:
                print(f"  [SKIP] no {L}-frame dump")
                g2 = False
                continue
            g2 &= check(f"{L}f AbsRel vs 8f (x{lim})", by_len[L]["absrel"],
                        "<=", a8 * lim)
        if 256 in by_len:
            g2 &= check("256f delta1 drop (pt)",
                        (d8 - by_len[256]["delta1"]) * 100, "<=", 3.0,
                        "{:.2f}")
    print("  [SKIP] late/early frame ratio and keyframe sawtooth amplitude: "
          "need per-frame traces (scripts/frame_index_probe.py)")

    print(f"\nP1 {'PASS' if p1 else 'FAIL'}   P2 {'PASS' if p2 else 'FAIL'}"
          f"   P3 {'PASS' if p3 else 'FAIL'}"
          f"   G1 {'PASS' if g1 else 'FAIL'}   G2 {'PASS' if g2 else 'FAIL'}")
    print("promotion needs P1, P2 and P3 (PLAN.md §3); G1/G2 are the older "
          "PLAN_ACC set, kept so a run between the two is visible")
    return 0 if (p1 and p2 and p3) else 1


if __name__ == "__main__":
    sys.exit(main())
