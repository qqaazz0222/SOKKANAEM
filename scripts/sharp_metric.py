"""Two numbers for the blur, because AbsRel does not see it.

Figure 9 shows our prediction with no object boundaries, yet its AbsRel is
competitive: a scene's error is dominated by the large smooth background, so a
silhouette can vanish entirely and barely move the average. Nothing reported in
this paper measures that, which is why the defect survived to the figure stage.

Two metrics, both per frame, both alignment-free by construction:

  edge AbsRel   AbsRel restricted to the depth-boundary band -- the pixels
                whose GT log-depth gradient is in the top decile of that
                frame, dilated by 3px. Same band `edge_weighted_loss` weights.

  gradient ratio  mean |grad of normalized-disparity(pred)| over the same
                band, divided by the GT's. This is the blur number: 1.0 means
                the prediction varies as fast as the truth does across a
                boundary, and 0.5 means it is twice as smooth. It is the
                spatial counterpart of Table 10's range ratio, and normalizing
                disparity per frame (MiDaS style, median-centre and
                mean-absolute-deviation scale) makes it free of scale and
                shift, so no alignment rule can flatter it.

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

D = "/home/hyunsu/dataset_ssd"
REAL = [f"tum:{D}/tum_static", f"bonn:{D}/bonn/rgbd_bonn_dataset"]
RHOLD = ["walking_static", "rgbd_bonn_crowd2", "rgbd_bonn_person_tracking2",
         "rgbd_bonn_static_close_far"]


def norm_disp(depth, valid):
    """Per-frame scale-shift normalized disparity (MiDaS). depth/valid (N,1,H,W).

    Per frame, not per batch: a batch holding one 0.5-129 m scene would
    otherwise set the scale for a 1.5-4 m one and flatten it to nothing."""
    d = 1.0 / depth.clamp(min=1e-3)
    out = torch.zeros_like(d)
    for i in range(d.shape[0]):
        v = valid[i].bool()
        if not bool(v.any()):
            continue
        t = d[i][v].median()
        s = (d[i][v] - t).abs().mean().clamp(min=1e-6)
        out[i] = (d[i] - t) / s
    return out


def grad_mag(x, valid):
    """|grad| and where it is defined, forward differences, same shape as x.

    Both endpoints of a difference must be valid. Without that, every sensor
    hole contributes a gradient the size of the hole's depth, the top decile
    of the GT gradient becomes the set of hole borders rather than the set of
    object boundaries, and the ratio below reads ~0 for any model."""
    vy = valid[..., 1:, :] * valid[..., :-1, :]
    vx = valid[..., :, 1:] * valid[..., :, :-1]
    gy = F.pad((x[..., 1:, :] - x[..., :-1, :]).abs() * vy, (0, 0, 0, 1))
    gx = F.pad((x[..., :, 1:] - x[..., :, :-1]).abs() * vx, (0, 1))
    defined = F.pad(vy, (0, 0, 0, 1)) * F.pad(vx, (0, 1))
    return gy + gx, defined


def boundary_band(gt, valid, decile=0.9, dilate=3):
    """Top-decile GT log-depth gradient, dilated. (N,1,H,W) bool."""
    g, defined = grad_mag(gt.clamp(min=1e-3).log(), valid)
    g = F.max_pool2d(g, dilate, stride=1, padding=dilate // 2) * defined
    band = torch.zeros_like(g, dtype=torch.bool)
    for i in range(g.shape[0]):
        d = defined[i].bool()
        if d.sum() < 10:
            continue
        thr = torch.quantile(g[i][d].float(), decile)
        band[i] = (g[i] >= thr) & d
    return band


def scale_shift(pred, gt, valid):
    """Least-squares scale+shift in disparity space, one fit per clip -- the
    MiDaS protocol, the native rule for the relative-depth baselines."""
    d = 1.0 / gt.clamp(min=1e-3)
    x, y = pred[valid].double(), d[valid].double()
    A = torch.stack([x, torch.ones_like(x)], 1)
    s, b = torch.linalg.lstsq(A, y.unsqueeze(1)).solution[:, 0]
    return 1.0 / (s * pred + b).clamp(min=1e-3)


@torch.no_grad()
def score_source(predict, loader, dev, max_clips):
    acc = {"absrel": [], "absrel_edge": [], "grad_ratio": []}
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

        rel = ((p - g).abs() / g.clamp(min=1e-3))
        acc["absrel"].append(float(rel[vb].mean()))

        band = boundary_band(g, v)
        if band.sum() > 0:
            acc["absrel_edge"].append(float(rel[band].mean()))
            dp, dg = norm_disp(p, v), norm_disp(g, v)
            gp, _ = grad_mag(dp, v)
            gg, _ = grad_mag(dg, v)
            acc["grad_ratio"].append(
                float(gp[band].mean() / gg[band].mean().clamp(min=1e-6)))
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
    def predict(clip, gt, valid):
        depths, _ = model.forward_clip(clip)      # detector-driven masks
        g = gt.reshape(-1, 1, *gt.shape[-2:])
        v = valid.reshape(-1, 1, *valid.shape[-2:]).bool()
        d = depths.reshape(-1, 1, *depths.shape[-2:])
        # median align per clip, the paper's rule for our metric model
        return d * (g[v].median() / d[v].median().clamp(min=1e-6))
    return predict, args.label or Path(args.ckpt).parent.name


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None, help="append one JSON line here")
    args = ap.parse_args()

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

    keys = ("absrel", "absrel_edge", "grad_ratio")
    rows["balanced"] = {k: sum(r[k] for r in rows.values()) / len(rows)
                        for k in keys}
    print(f"\n{label}")
    print(f"{'source':>10s} " + " ".join(f"{k:>12s}" for k in keys))
    for name, r in rows.items():
        print(f"{name:>10s} " + " ".join(f"{r[k]:>12.4f}" for k in keys))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps({"label": label, "ckpt": args.ckpt,
                                 "scores": rows}) + "\n")


if __name__ == "__main__":
    main()
