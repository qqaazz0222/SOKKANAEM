"""Figure 8: the qualitative comparison, one row per scene type.

Columns: RGB | activity mask | ours | a comparable-size baseline | a large
baseline | ground truth. The mask column does work no other figure does --
it shows the detector firing on the moving object and nowhere else -- which
is why it sits next to the RGB rather than at the end.

Every model is aligned by its own native rule, the same convention Table 3
uses: per-clip median scaling for ours (metric), least-squares scale+shift in
disparity space for the relative baselines. Aligning them all one way would
flatter one family and penalise the other (Section 5.4).

Colour range is per row, taken from that row's GT inverse depth, and shared
by every depth tile in the row -- a per-tile range would make a wrong
prediction look right.

    python scripts/viz_qualitative.py --out paper/figures/qualitative.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sokkanaem import from_checkpoint
from sokkanaem.data import build_mixed
from viz import colorize

D = "/home/hyunsu/dataset_ssd"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# (row label, dataset spec, holdout sequences, which clip of the split)
SCENES = [
    ("Static indoor", f"tum:{D}/tum_static", ["walking_static"], None),
    ("Moving person", f"bonn:{D}/bonn/rgbd_bonn_dataset",
     ["rgbd_bonn_person_tracking2"], None),
    ("Driving", f"kitti:{D}/kitti_zs", None, None),
]
BASELINES = [
    ("DA V2 Small", "depth-anything/Depth-Anything-V2-Small-hf"),
    ("DPT-Large", "Intel/dpt-large"),
]
COLS = ["RGB", "Activity mask", "Ours", None, None, "Ground truth"]


def scale_shift(pred, gt, valid):
    """Least-squares scale+shift in disparity space, one fit per clip.

    The MiDaS protocol for relative-depth models; returns aligned depth."""
    disp_gt = 1.0 / np.clip(gt, 1e-3, None)
    x, y = pred[valid].astype(np.float64), disp_gt[valid].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return 1.0 / np.clip(s * pred + b, 1e-3, None)


def mask_overlay(rgb, mask, grid):
    """RGB with skipped patches dimmed and active ones left bright.

    Dimming the skipped side rather than tinting the active side keeps the
    scene legible: the reader sees what the model looked at, in context."""
    m = mask.reshape(grid, grid)
    up = np.kron(m, np.ones((rgb.shape[0] // grid, rgb.shape[1] // grid)))
    up = up[..., None]
    dim = (rgb * 0.28).astype(np.uint8)
    out = np.where(up > 0.5, rgb, dim).astype(np.uint8)
    # thin outline on the active/skipped boundary, so the patch grid reads
    # as a decision rather than as a lighting effect
    edge = np.abs(np.diff(up[:, :, 0], axis=0, prepend=up[:1, :, 0])) + \
        np.abs(np.diff(up[:, :, 0], axis=1, prepend=up[:, :1, 0]))
    out[edge > 0.5] = np.array([255, 214, 10], np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/v11-longclip-spread-s0/latest.pt")
    ap.add_argument("--out", default="paper/figures/qualitative.png")
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--clips", type=int, nargs="+", default=None,
                    help="one clip index per row; default picks the middle")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = from_checkpoint(args.ckpt, dev).eval()

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    bl = []
    for label, hf in BASELINES:
        proc = AutoImageProcessor.from_pretrained(hf)
        net = AutoModelForDepthEstimation.from_pretrained(hf).to(dev).eval()
        bl.append((label, proc, net))

    rows, notes = [], []
    for ri, (label, spec, holdout, forced) in enumerate(SCENES):
        ds, _ = build_mixed([spec], clip_len=args.clip_len,
                            clip_stride=args.clip_len, size=args.size,
                            holdout=holdout, val=True)
        ci = forced if forced is not None else (
            args.clips[ri] if args.clips else len(ds) // 2)
        clip, gt, valid = ds[int(ci)]
        gtn = gt.numpy()[:, 0]
        vn = valid.numpy()[:, 0] > 0
        rgb = (clip.numpy()[-1].transpose(1, 2, 0) * 255).astype(np.uint8)

        # ours: streaming over the whole clip so the state and the mask are
        # warmed up exactly as they would be at deployment.
        #
        # Two passes, because the mask column and the depth column answer
        # different questions. Above dense_above the model replaces the mask
        # with all-ones and pays for a dense frame, so info["mask"] from the
        # shipped configuration shows 100% active and hides the very decision
        # this column exists to show -- that is what sank the earlier attempt
        # at a qualitative panel. The detector state is unaffected by the
        # override (model.py only rewrites `mask`, never `det`), so the
        # detector pass yields exactly the selection the shipped model made
        # before deciding to overrule it.
        def stream(dense_above):
            model.dense_above = dense_above
            state, ms = None, []
            with torch.no_grad():
                for t in range(clip.shape[0]):
                    d, state, info = model.step(clip[None, t].to(dev), state)
                    ms.append(info["mask"][0].cpu().numpy())
            return d, ms

        shipped = model.dense_above
        _, det_masks = stream(0.0)
        depth, paid_masks = stream(shipped)
        ours = depth[0, 0].cpu().numpy()
        v = vn[-1]
        ours = ours * (np.median(gtn[-1][v]) / max(np.median(ours[v]), 1e-6))
        masks = det_masks
        grid = int(round(np.sqrt(masks[-1].size)))
        active = float(np.mean([m.mean() for m in masks[1:]]))
        paid = float(np.mean([m.mean() for m in paid_masks[1:]]))

        # baselines: per-frame, then one scale+shift over the whole clip
        preds = []
        for blabel, proc, net in bl:
            out = []
            with torch.no_grad():
                for t in range(clip.shape[0]):
                    img = Image.fromarray(
                        (clip[t].permute(1, 2, 0).numpy() * 255).astype("uint8"))
                    inp = proc(images=img, return_tensors="pt").to(dev)
                    post = proc.post_process_depth_estimation(
                        net(**inp), target_sizes=[clip.shape[-2:]])
                    out.append(post[0]["predicted_depth"].cpu().numpy())
            out = np.stack(out)
            preds.append(scale_shift(out, gtn, vn)[-1])

        gt_disp = np.where(v, 1.0 / np.clip(gtn[-1], 1e-3, None), 0.0)
        lo, hi = np.percentile(gt_disp[v], [1, 99])
        tiles = [rgb, mask_overlay(rgb, masks[-1], grid),
                 colorize(1.0 / np.clip(ours, 1e-3, None), lo, hi)]
        tiles += [colorize(1.0 / np.clip(p, 1e-3, None), lo, hi) for p in preds]
        tiles.append(np.where(v[..., None], colorize(gt_disp, lo, hi), 0))
        rows.append((label, active, tiles))
        notes.append(f"{label}: clip {ci}/{len(ds)}, detector {active*100:.1f}%, "
                     f"paid {paid*100:.1f}%, GT valid {v.mean()*100:.1f}%")
        print(notes[-1])

    # compose
    S, gap, lab_w, hdr_h = args.size, 6, 132, 22
    ncol = len(rows[0][2])
    W = lab_w + ncol * S + (ncol - 1) * gap
    H = hdr_h + len(rows) * S + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(FONT, 13)
    fs = ImageFont.truetype(FONT, 11)
    heads = list(COLS)
    heads[3], heads[4] = BASELINES[0][0], BASELINES[1][0]
    for c, h in enumerate(heads):
        x = lab_w + c * (S + gap)
        dr.text((x + S // 2, hdr_h - 6), h, fill="black", font=f, anchor="ms")
    for r, (label, active, tiles) in enumerate(rows):
        y = hdr_h + r * (S + gap)
        dr.text((lab_w - 10, y + S // 2 - 7), label, fill="black", font=f,
                anchor="rs")
        dr.text((lab_w - 10, y + S // 2 + 9), f"detector {active*100:.0f}%",
                fill="#555555", font=fs, anchor="rs")
        for c, t in enumerate(tiles):
            canvas.paste(Image.fromarray(t), (lab_w + c * (S + gap), y))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"\n-> {args.out}  ({W}x{H})")


if __name__ == "__main__":
    main()
