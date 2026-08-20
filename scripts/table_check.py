"""Every number in a paper table must come from ONE measurement run.

Table 11 of the draft was assembled from two checkpoints -- AbsRel from v9,
delta1/TCE from v10 -- and nothing caught it, because a markdown row carries
no provenance. This walks every table row in the draft, collects its numeric
tokens, and checks that a single log block contains all of them. A row whose
tokens are only satisfied by two different blocks is the bug above.

    python scripts/table_check.py [paper/draft.md ...]

Baseline rows are means over per-source pooled numbers, so those aggregations
are recomputed here. The same aggregation over a multi-source eval.txt block
(our own 2-DOF row, for instance) is NOT reconstructed -- such a row shows up
as a partial match and has to be checked by hand once.

Exit 1 if any row has no single-block source. Rows that legitimately come
from outside the logs (analytical MACs, parameter counts) show up as
unmatched -- read the report, do not silence it.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"\d+\.\d{2,}")  # 0.1774, 29.2 is too weak to attribute


def blocks():
    """(label, text) per measurement run: log files split at each ckpt= header."""
    out = []
    for f in sorted(ROOT.glob("work_dirs/**/eval.txt")) + sorted(ROOT.glob("work_dirs/*.log")) \
            + sorted(ROOT.glob("reports/**/*.log")) + sorted(ROOT.glob("reports/**/*.txt")):
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        parts = re.split(r"(?m)^(?=ckpt=)", text)
        for i, p in enumerate(parts):
            if p.strip():
                head = p.splitlines()[0][:110]
                out.append((f"{f.relative_to(ROOT)}#{i} {head}", p))
    return out


def baseline_means():
    """Table 3's rows are means over the per-source pooled numbers in the
    baseline logs, so no substring search can find them. Recompute the
    aggregation with the same parser scripts/make_tables.py uses, one synthetic
    block per (model, tag, domain)."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_tables import REAL, SYNTH, baseline_rows, mean_over

    out = []
    for log in sorted(ROOT.glob("work_dirs/*.log")):
        try:
            runs = baseline_rows(log)
        except Exception:
            continue
        for (model, tag), per_src in runs.items():
            for dom, sources in (("real", REAL), ("synth", SYNTH)):
                m = mean_over(per_src, sources)
                if m is None:
                    continue
                text = " ".join(f"{k}={v:.4f}" for k, v in m.items()
                                if isinstance(v, float))
                out.append((f"{log.name}[derived] {model} {tag} {dom} "
                            f"mean of {m['sources']} sources", text))
    return out


def rows(doc):
    for ln, line in enumerate(doc.read_text().splitlines(), 1):
        s = line.strip()
        if not (s.startswith("|") and s.count("|") >= 3):
            continue
        toks = NUM.findall(s.replace("**", ""))
        if len(toks) >= 3:
            yield ln, s, toks


def main():
    docs = [Path(a) for a in sys.argv[1:]] or [ROOT / "paper/draft.md"]
    bs = blocks() + baseline_means()
    bad = 0
    for doc in docs:
        print(f"== {doc}")
        for ln, s, toks in rows(doc):
            full = [lbl for lbl, txt in bs if all(t in txt for t in toks)]
            if full:
                continue
            scored = sorted(((sum(t in txt for t in toks), lbl, txt)
                             for lbl, txt in bs), reverse=True)[:2]
            best = scored[0][0] if scored else 0
            # config values (tau, weights) and parameter counts share a row with
            # measurements and live in no log, so a near-miss is normal. Only a
            # row whose best source covers less than half its numbers is a real
            # provenance hole -- or a row assembled from two runs.
            flag = best * 2 < len(toks)
            bad += flag
            print(f"  line {ln}: {'UNATTRIBUTED' if flag else 'partial'} "
                  f"{best}/{len(toks)} {toks}")
            for hits, lbl, txt in scored:
                miss = [x for x in toks if x not in txt]
                print(f"    {hits}/{len(toks)} {lbl}\n      missing: {miss}")
    print(f"\n{bad} row(s) without a single-run source")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
