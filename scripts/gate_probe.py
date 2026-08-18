"""What does the change detector actually see, before any deployment policy?

`infer.py` reports the compute that was paid, so a frame routed to the dense
fallback (`dense_above`) shows up as 100% active whatever the detector said.
That is the right number for a latency claim and the wrong one for comparing
*gating strategies* -- which is what T3-14 needs: pixel gating and GMC +
feature gating live on different scales (MSE vs relative feature L1), so the
only fair comparison is the active-ratio curve each traces as tau sweeps.

Reports the raw detector ratio with the fallback disabled.
"""
import argparse
import glob
import os

import torch
from PIL import Image

from sokkanaem import checkpoint_config, from_checkpoint


def load_frames(d, n, size):
    paths = sorted(glob.glob(os.path.join(d, "*.png"))
                   + glob.glob(os.path.join(d, "*.jpg")))[:n]
    assert paths, f"no frames in {d}"
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        out.append(torch.from_numpy(
            __import__("numpy").asarray(im)).permute(2, 0, 1).float() / 255)
    return torch.stack(out)


@torch.no_grad()
def sweep(ckpt, frames, gmc, taus, dev):
    rows = []
    for tau in taus:
        model = from_checkpoint(ckpt, dev, gmc=gmc, tau_on=tau,
                                tau_off=tau / 2).eval()
        model.dense_above = 0.0          # the point: measure the detector
        state, ratios = None, []
        for i in range(frames.shape[0]):
            _, state, info = model.step(frames[i:i + 1].to(dev), state)
            if i:                        # frame 0 is always a keyframe
                ratios.append(info["active_ratio"])
        rows.append((tau, sum(ratios) / len(ratios)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames-dir", required=True, nargs="+")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--pixel-tau", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--gmc-tau", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.2, 0.4, 0.8])
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    size = args.size or checkpoint_config(args.ckpt).get("size", 256)

    for d in args.frames_dir:
        frames = load_frames(d, args.frames, size)
        print(f"\n=== {os.path.basename(os.path.dirname(os.path.dirname(d)))} "
              f"({frames.shape[0]} frames @ {size}px) ===")
        pix = sweep(args.ckpt, frames, False, args.pixel_tau, dev)
        gmc = sweep(args.ckpt, frames, True, args.gmc_tau, dev)
        print(f"{'pixel tau':>10s} {'active%':>8s}    {'GMC tau':>8s} {'active%':>8s}")
        for i in range(max(len(pix), len(gmc))):
            a = f"{pix[i][0]:10g} {pix[i][1]*100:7.1f}" if i < len(pix) else " " * 18
            b = f"{gmc[i][0]:8g} {gmc[i][1]*100:7.1f}" if i < len(gmc) else ""
            print(f"{a}    {b}")


if __name__ == "__main__":
    main()
