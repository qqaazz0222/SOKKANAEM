"""Read an arm's numbers and pick the next experiment, with the rule written
down before the result is known.

The rules below are the ones stated when the arm was launched (REPORT §4.46):
the M0 student exists to answer whether the Q0 teacher is absorbable at 10.8M,
and each answer has a different next move. Encoding it here rather than
deciding after the fact keeps the choice from drifting to fit whatever came
out.

    python scripts/decide_next.py m0-distil-s0 --launch
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

# the control this arm is judged against: same recipe, no teacher, 4.19M
BASE = {"absrel": 0.1234, "grad_ratio": 0.4310, "boundary_f1": 0.3417}
Q0 = {"grad_ratio": 0.7095, "boundary_f1": 0.6191}


def screening(label, path="outputs/sharpness/suite.jsonl"):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["label"]] = r["scores"]["balanced"]
    return rows.get(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--launch", action="store_true",
                    help="actually start the branch instead of printing it")
    args = ap.parse_args()

    s = screening(args.arm)
    if s is None:
        print(f"no sharpness row for {args.arm}", file=sys.stderr)
        return 1
    grad, f1, absrel = s["grad_ratio"], s["boundary_f1"], s["absrel"]
    print(f"{args.arm}: AbsRel {absrel:.4f} (base {BASE['absrel']:.4f})  "
          f"grad {grad:.4f} (base {BASE['grad_ratio']:.4f}, Q0 {Q0['grad_ratio']:.4f})  "
          f"F1 {f1:.4f}")

    if grad > 1.0:
        # PLAN §3.5's third revision, which this rule failed to encode: a
        # gradient ratio above the ground truth's own is oversharpening, not
        # sharpness. The 60k distillation arm read 1.80 with flat TV 0.16 and
        # overshoot 0.76, and the branch below would have called that progress
        # and spent another 30 GPU-hours widening it.
        branch, why = None, (
            f"grad_ratio {grad:.3f} is ABOVE the ground truth's -- the arm is "
            f"oversharpened, not sharp (read flat TV and overshoot with it). "
            f"No branch: lower the gradient-pushing weights or add the "
            f"overshoot penalty before spending more")
        print(f"decision: {why}")
        return 0
    if grad >= 0.65 and absrel <= BASE["absrel"]:
        # the teacher is absorbable and capacity was the limit: spend it
        branch, why = "work_dirs/queue19.sh", (
            "teacher absorbed at 10.8M -- scale the student to 13.6M and the "
            "distillation to 24k steps")
    elif grad >= BASE["grad_ratio"] + 0.05:
        # it moves sharpness but not enough: more teacher signal, not more model
        branch, why = "work_dirs/queue20.sh", (
            "teacher helps but does not close the gap -- widen the distillation "
            "data instead of the student")
    else:
        branch, why = None, (
            "the student cannot absorb the teacher at this size either. The "
            "native track is done; Q0 is the quality candidate and the "
            "remaining effort belongs on streaming (queue18) and the write-up")
    print(f"decision: {why}")
    if branch and args.launch:
        subprocess.Popen(["nohup", "bash", branch],
                         stdout=open(f"{branch[:-3]}.log", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
        print(f"launched {branch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
