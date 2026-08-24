"""Why is our depth column a blob? Three suspects, one picture.

The qualitative figure (Figure 9) shows our prediction with no object
boundaries at all -- far beyond the range compression of Section 6.5, which
flattens a field but does not erase a silhouette. Three explanations were on
the table and they imply opposite next steps:

  patch grid   -> the 16x16 token output cannot hold a silhouette
  input res    -> we run at 256 while the baselines' own processors upsample
                  to their native 518 / 384, i.e. ~5x the tokens
  the model    -> grid and resolution are fine and the network is not using
                  them

Columns, per scene row:

    RGB | ours 256 | ours 512 | ours 256, RGB skips zeroed
        | oracle p16 | oracle p8 | DA V2 Small | GT

`oracle p16` is Section 6.4's ceiling as a picture: ground truth pushed
through our own output bottleneck (avg-pool to the token grid over valid
pixels, bilinear back up, `ceiling_probe.through_grid`). Whatever structure
that tile holds is structure the patch grid CAN represent, so if it is sharp
and ours is not, patch size is exonerated. `oracle p8` is what a finer grid
would buy.

`RGB skips zeroed` tests the decoder: DPTDecoder's only source of detail
finer than a patch is a 3-layer stride-2 conv stem on the frame. If zeroing
it changes nothing, the decoder ignores its one detail path.

Every depth tile in a row shares one colour range taken from that row's GT,
and ours is median-aligned per clip -- the conventions of Figure 9.

    python scripts/sharpness_probe.py --out outputs/sharpness/probe.png
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
from ceiling_probe import through_grid
from viz import colorize
from viz_qualitative import FONT, SCENES, scale_shift

BASELINE = ("DA V2 Small", "depth-anything/Depth-Anything-V2-Small-hf")
COLS = ["RGB", "Ours 256", "Ours 512", "Ours, no RGB skip",
        "Oracle p16", "Oracle p8", BASELINE[0], "Ground truth"]


def stream_last(model, clip, dev):
    """Streaming pass over the whole clip; returns the last frame's depth,
    so state and mask are warmed exactly as at deployment (viz_qualitative)."""
    state = None
    with torch.no_grad():
        for t in range(clip.shape[0]):
            d, state, _ = model.step(clip[None, t].to(dev), state)
    return d[0, 0].float().cpu().numpy()


def median_align(pred, gt, valid):
    return pred * (np.median(gt[valid]) / max(np.median(pred[valid]), 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/v11-longclip-spread-s0/latest.pt")
    ap.add_argument("--out", default="outputs/sharpness/probe.png")
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--big", type=int, default=512, help="high-res arm")
    ap.add_argument("--tile", type=int, default=256, help="tile size in the PNG")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = from_checkpoint(args.ckpt, dev).eval()

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(BASELINE[1])
    net = AutoModelForDepthEstimation.from_pretrained(BASELINE[1]).to(dev).eval()

    def load(spec, holdout, size):
        ds, _ = build_mixed([spec], clip_len=args.clip_len,
                            clip_stride=args.clip_len, size=size,
                            holdout=holdout, val=True)
        return ds

    rows = []
    for label, spec, holdout, forced in SCENES:
        ds = load(spec, holdout, args.size)
        ci = forced if forced is not None else len(ds) // 2
        clip, gt, valid = ds[int(ci)]
        gtn, vn = gt.numpy()[:, 0], valid.numpy()[:, 0] > 0
        v = vn[-1]
        rgb = (clip.numpy()[-1].transpose(1, 2, 0) * 255).astype(np.uint8)

        ours = median_align(stream_last(model, clip, dev), gtn[-1], v)

        # high-res arm: same clip index, the loader's only difference is the
        # resize, so the frames are the same footage at 2x the token grid
        ds_b = load(spec, holdout, args.big)
        clip_b, gt_b, valid_b = ds_b[int(ci)]
        big = stream_last(model, clip_b, dev)
        vb = valid_b.numpy()[-1, 0] > 0
        big = median_align(big, gt_b.numpy()[-1, 0], vb)
        big = np.asarray(Image.fromarray(big.astype(np.float32)).resize(
            (ours.shape[1], ours.shape[0]), Image.BILINEAR))
        big = median_align(big, gtn[-1], v)   # re-align at the common 256 grid

        # decoder ablation: kill the only sub-patch detail path
        hooks = [b.register_forward_hook(lambda m, i, o: o * 0)
                 for b in model.decoder.stem]
        noskip = median_align(stream_last(model, clip, dev), gtn[-1], v)
        for h in hooks:
            h.remove()

        # Section 6.4's ceiling, as pictures
        gd, vd = gt.to(dev), valid.to(dev)          # (T, 1, H, W)
        oracles = [through_grid(gd, vd, p, True)[-1, 0].float().cpu().numpy()
                   for p in (16, 8)]

        # baseline, its own native rule: per-frame, one scale+shift per clip
        out = []
        with torch.no_grad():
            for t in range(clip.shape[0]):
                img = Image.fromarray(
                    (clip[t].permute(1, 2, 0).numpy() * 255).astype("uint8"))
                inp = proc(images=img, return_tensors="pt").to(dev)
                post = proc.post_process_depth_estimation(
                    net(**inp), target_sizes=[clip.shape[-2:]])
                out.append(post[0]["predicted_depth"].float().cpu().numpy())
        da2 = scale_shift(np.stack(out), gtn, vn)[-1]

        gt_disp = np.where(v, 1.0 / np.clip(gtn[-1], 1e-3, None), 0.0)
        lo, hi = np.percentile(gt_disp[v], [1, 99])
        c = lambda x: colorize(1.0 / np.clip(x, 1e-3, None), lo, hi)
        tiles = [rgb, c(ours), c(big), c(noskip), c(oracles[0]), c(oracles[1]),
                 c(da2), np.where(v[..., None], colorize(gt_disp, lo, hi), 0)]
        rows.append((label, tiles))

        # a number for each column, so the picture is not the only evidence:
        # AbsRel of that tile against GT on valid pixels
        def absrel(x):
            return float(np.mean(np.abs(x[v] - gtn[-1][v]) / gtn[-1][v]))
        print(f"{label:>14s}  ours {absrel(ours):.4f}  512 {absrel(big):.4f}  "
              f"noskip {absrel(noskip):.4f}  p16 {absrel(oracles[0]):.4f}  "
              f"p8 {absrel(oracles[1]):.4f}  da2 {absrel(da2):.4f}")

    S, gap, lab_w, hdr_h = args.tile, 6, 132, 22
    ncol = len(COLS)
    W = lab_w + ncol * S + (ncol - 1) * gap
    H = hdr_h + len(rows) * S + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(FONT, 13)
    for ci_, h in enumerate(COLS):
        dr.text((lab_w + ci_ * (S + gap) + S // 2, hdr_h - 6), h,
                fill="black", font=f, anchor="ms")
    for r, (label, tiles) in enumerate(rows):
        y = hdr_h + r * (S + gap)
        dr.text((lab_w - 10, y + S // 2), label, fill="black", font=f,
                anchor="rs")
        for ci_, t in enumerate(tiles):
            im = Image.fromarray(t.astype(np.uint8))
            if im.size != (S, S):
                im = im.resize((S, S), Image.NEAREST)
            canvas.paste(im, (lab_w + ci_ * (S + gap), y))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"\n-> {args.out}  ({W}x{H})")


if __name__ == "__main__":
    main()
