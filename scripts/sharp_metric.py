"""The sharpness suite, because AbsRel does not see the blur.

Figure 9 shows our prediction with no object boundaries, yet its AbsRel is
competitive: a scene's error is dominated by the large smooth background, so a
silhouette can vanish entirely and barely move the average. Nothing reported in
this paper measured that, which is why the defect survived to the figure stage.

The metrics live in `sokkanaem/sharpness.py` (tests/test_sharpness.py pins
their directions); this script runs them on the eval.py protocol. All are per
frame on scale-shift-normalized disparity, so no alignment rule flatters them:

  absrel_edge     AbsRel restricted to the depth-boundary band -- the pixels
                  whose GT log-depth gradient is in the top decile of the
                  frame, dilated by 3px. Same band `edge_weighted_loss` weights.
  grad_ratio      the blur number: prediction gradient over GT gradient in
                  that band. 1.0 varies as fast as the truth, 0.5 is twice as
                  smooth. Buyable with noise, hence the next three.
  boundary_*      precision/recall/F1 of the boundary set at the GT's own
                  0.90/0.95/0.99 gradient quantiles, 2px matching slack:
                  blur loses recall, noise loses precision.
  flat_tv[_ratio] prediction TV inside GT-flat regions, and over the GT's --
                  the noise term.
  overshoot       share of band pixels leaving the local GT range by >5% of
                  it -- the ringing term.

Reported per source and dataset-balanced, on the eval.py protocol: clip-len 8,
100 clips spread evenly over each holdout.

    python scripts/sharp_metric.py --ckpt work_dirs/<run>/latest.pt
"""
import argparse
import json
from pathlib import Path

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ceiling_probe import through_grid
from sokkanaem import from_checkpoint
from sokkanaem.data import build_mixed, even_subset
from sokkanaem.losses import dynamic_mask
from sokkanaem.sharpness import sharpness_scores

D = "/home/hyunsu/dataset_ssd"
REAL = [f"tum:{D}/tum_static", f"bonn:{D}/bonn/rgbd_bonn_dataset"]
RHOLD = ["walking_static", "rgbd_bonn_crowd2", "rgbd_bonn_person_tracking2",
         "rgbd_bonn_static_close_far"]


def scale_shift(pred, gt, valid):
    """Least-squares scale+shift in disparity space, one fit per clip -- the
    MiDaS protocol, the native rule for the relative-depth baselines."""
    d = 1.0 / gt.clamp(min=1e-3)
    x, y = pred[valid].double(), d[valid].double()
    A = torch.stack([x, torch.ones_like(x)], 1)
    s, b = torch.linalg.lstsq(A, y.unsqueeze(1)).solution[:, 0]
    return 1.0 / (s * pred + b).clamp(min=1e-3)


KEYS = ("absrel", "absrel_dyn", "absrel_near", "absrel_edge", "grad_ratio",
        "boundary_f1", "boundary_precision", "boundary_recall", "flat_tv",
        "flat_tv_gt", "flat_tv_ratio", "overshoot")


@torch.no_grad()
def score_source(predict, loader, dev, max_clips):
    acc = {k: [] for k in KEYS}
    for ci, (clip, gt, valid) in enumerate(loader):
        if ci >= max_clips:
            break
        clip, gt, valid = clip.to(dev), gt.to(dev), valid.to(dev)
        B, T = gt.shape[:2]
        f = lambda x: x.reshape(B * T, 1, *x.shape[-2:])
        g, v = f(gt), f(valid)
        vb = v.bool()
        if not bool(vb.any()):
            continue
        p = f(predict(clip, gt, valid))

        rel = (p - g).abs() / g.clamp(min=1e-3)
        acc["absrel"].append(float(rel[vb].mean()))
        near = vb & (g < 2.0)          # the 2.38x gap (REPORT 4.43)
        if near.any():
            acc["absrel_near"].append(float(rel[near].mean()))
        # the 2.66x gap, and the probe's Go criterion. Same definition as
        # eval_acc's `dynamic` region and dynamic_weighted_loss's mask: flow
        # that disagrees with the frame's dominant motion, so a panning camera
        # does not mark the whole scene moving.
        if min(clip.shape[-2:]) >= 128:
            dyn = f(dynamic_mask(clip).float()).bool() & vb
            if dyn.any():
                acc["absrel_dyn"].append(float(rel[dyn].mean()))
        for k, x in sharpness_scores(p, g, v).items():
            if k in acc and x == x:          # drop NaN (no boundary in frame)
                acc[k].append(x)
    return {k: sum(x) / max(len(x), 1) for k, x in acc.items()}


def build_predictor(args, dev):
    """Returns (predict, label). predict(clip, gt, valid) -> depths (B,T,1,H,W),
    each arm aligned by its own native rule, as Table 3 and Figure 9 do."""
    if args.oracle_patch:
        p = args.oracle_patch
        def predict(clip, gt, valid):
            B, T = gt.shape[:2]
            f = lambda x: x.reshape(B * T, 1, *x.shape[-2:])
            return through_grid(f(gt), f(valid), p, True)   # already metric
        return predict, args.label or f"oracle p{p}"
    if args.baseline:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        proc = AutoImageProcessor.from_pretrained(args.baseline)
        net = AutoModelForDepthEstimation.from_pretrained(args.baseline).to(dev).eval()
        def predict(clip, gt, valid):
            B, T = clip.shape[:2]
            flat = clip.reshape(B * T, *clip.shape[-3:])
            out = []
            for i in range(flat.shape[0]):
                inp = proc(images=flat[i], return_tensors="pt", do_rescale=False)
                inp = {k: v.to(dev) for k, v in inp.items()}
                post = proc.post_process_depth_estimation(
                    net(**inp), target_sizes=[clip.shape[-2:]])
                out.append(post[0]["predicted_depth"].float())
            d = torch.stack(out).unsqueeze(1)                # (B*T, 1, H, W)
            g = gt.reshape(-1, 1, *gt.shape[-2:])
            v = valid.reshape(-1, 1, *valid.shape[-2:]).bool()
            return scale_shift(d, g, v)
        return predict, args.label or args.baseline.split("/")[-1]
    model = from_checkpoint(args.ckpt, dev).eval()
    if args.tau is not None:
        # tau=0 activates every patch, i.e. the dense path. A model trained
        # dense (the Stage A probes) scored through its detector is being read
        # off-distribution, and the two capacity arms were not off-distribution
        # by the same amount -- which is how M0 came out with a LOWER training
        # loss and WORSE fit numbers than the 4.19M arm.
        model.detector.tau_on, model.detector.tau_off = args.tau, args.tau / 2
    def predict(clip, gt, valid):
        depths, _ = model.forward_clip(clip)      # detector-driven masks
        g = gt.reshape(-1, 1, *gt.shape[-2:])
        v = valid.reshape(-1, 1, *valid.shape[-2:]).bool()
        d = depths.reshape(-1, 1, *depths.shape[-2:])
        # median align per clip, the paper's rule for our metric model
        return d * (g[v].median() / d[v].median().clamp(min=1e-6))
    return predict, args.label or Path(args.ckpt).parent.name


def table(path):
    """Every arm scored into one JSONL, as one comparison table. Later lines
    win, so re-scoring an arm replaces its row rather than adding a second."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["label"]] = r["scores"]["balanced"]
    keys = [k for k in KEYS if any(k in v for v in rows.values())]
    print(f"{'arm':<28}" + "".join(f"{k:>19}" for k in keys))
    for label, sc in rows.items():
        print(f"{label:<28}" + "".join(
            f"{sc.get(k, float('nan')):>19.4f}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=None,
                    help="print every arm in this JSONL as one table and exit")
    ap.add_argument("--ckpt", default=None, help="our checkpoint")
    ap.add_argument("--baseline", default=None,
                    help="HF depth model id, scored under its own native "
                         "scale+shift rule (e.g. "
                         "depth-anything/Depth-Anything-V2-Small-hf)")
    ap.add_argument("--oracle-patch", type=int, default=None,
                    help="score GT pushed through a patch-P token grid "
                         "instead of a model -- Section 6.4's ceiling, on "
                         "these two metrics")
    ap.add_argument("--data", nargs="+", default=REAL)
    ap.add_argument("--holdout", nargs="+", default=RHOLD)
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-clips", type=int, default=100)
    ap.add_argument("--tau", type=float, default=None,
                    help="ours: override the detector threshold (0 = every "
                         "patch active, the dense path)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None, help="append one JSON line here")
    args = ap.parse_args()
    if args.table:
        return table(args.table)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    predict, label = build_predictor(args, dev)
    dataset, _ = build_mixed(args.data, clip_len=args.clip_len,
                             clip_stride=args.clip_len, size=args.size,
                             holdout=args.holdout, val=True)

    rows = {}
    for spec, ds in zip(args.data, dataset.datasets):
        name = spec.split(":")[0]
        loader = torch.utils.data.DataLoader(
            even_subset(ds, args.max_clips), batch_size=1, shuffle=False)
        rows[name] = score_source(predict, loader, dev, args.max_clips)

    keys = KEYS
    rows["balanced"] = {k: sum(r[k] for r in rows.values()) / len(rows)
                        for k in keys}
    print(f"\n{label}")
    print(f"{'source':>10s} " + " ".join(f"{k:>12s}" for k in keys))
    for name, r in rows.items():
        print(f"{name:>10s} " + " ".join(f"{r[k]:>12.4f}" for k in keys))
    print("\ngrad_ratio alone is buyable: read it with boundary_f1 (blur "
          "loses recall, noise loses precision), flat_tv_ratio (noise) and "
          "overshoot (ringing).")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps({"label": label, "ckpt": args.ckpt,
                                 "scores": rows}) + "\n")


if __name__ == "__main__":
    main()
