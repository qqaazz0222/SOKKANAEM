"""Streaming inference: per-frame active ratio report + optional depth PNGs.

Input: --video file (needs opencv), --frames-dir (folder of images),
or synthetic sequence by default.

Run:
    python scripts/infer.py --ckpt sokkanaem.pt
    python scripts/infer.py --ckpt sokkanaem.pt --video cam.mp4 --save-dir out/
    python scripts/infer.py --ckpt sokkanaem.pt --frames-dir data/seq0/rgb --save-dir out/
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from sokkanaem import SOKKANAEM, checkpoint_config, from_checkpoint
from sokkanaem.data import SynthClips


def frames_from_video(path, size):
    import cv2
    cap = cv2.VideoCapture(path)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size))
        yield torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def frames_from_dir(path, size):
    for p in sorted(glob.glob(os.path.join(path, "*"))):
        img = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        yield torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255


def save_depth(depth, path):
    """16-bit PNG, millimeters — same convention the loaders read back."""
    d = (depth[0, 0].cpu().numpy() * 1000).clip(0, 65535).astype(np.uint16)
    Image.fromarray(d).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--save-dir", default=None,
                    help="depth PNG output dir; default <ckpt dir>/viz when "
                         "--ckpt given, 'none' disables")
    ap.add_argument("--size", type=int, default=None,
                    help="inference resolution; default = the checkpoint's "
                         "training size (128 without a checkpoint)")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--gmc", action=argparse.BooleanOptionalAction, default=None,
                    help="ego-motion mode: Low-Res GMC + feature gating (§3.5)")
    ap.add_argument("--tau-on", type=float, default=None,
                    help="gate threshold override (feature scale with --gmc)")
    ap.add_argument("--tau-off", type=float, default=None)
    # tri-state on purpose: a plain store_true default of False overrode the
    # checkpoint's trained spatial_cache=true, so the default run silently took
    # the dense path (89.8% of full MACs instead of 38.9%)
    ap.add_argument("--spatial-cache", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="reuse static-patch spatial outputs (§4.5 wall-clock); "
                         "default = whatever the checkpoint was trained with")
    ap.add_argument("--temporal-cache", action=argparse.BooleanOptionalAction,
                    default=None, help="reuse static-patch temporal readout")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = checkpoint_config(args.ckpt) if args.ckpt else {}
    if args.size is None:
        args.size = cfg.get("size", 128)
    kw = {k: v for k, v in (("gmc", args.gmc),
                            ("spatial_cache", args.spatial_cache),
                            ("temporal_cache", args.temporal_cache))
          if v is not None}
    if args.gmc:  # feature-scale defaults (relative L1), not pixel MSE
        kw.update(tau_on=0.1, tau_off=0.05)
    if args.tau_on is not None:
        kw["tau_on"] = args.tau_on
    if args.tau_off is not None:
        kw["tau_off"] = args.tau_off
    if args.ckpt:
        # trained [model] kwargs come from config.toml next to the ckpt
        model = from_checkpoint(args.ckpt, dev, **kw).eval()
    else:
        model = SOKKANAEM(**kw).to(dev).eval()
    print(f"size {args.size}  spatial_cache {model.spatial_cache}  "
          f"temporal_cache {model.temporal_cache}  "
          f"tau_on {model.detector.tau_on:g}")
    if args.save_dir is None and args.ckpt:
        args.save_dir = os.path.join(os.path.dirname(args.ckpt) or ".", "viz")
    if args.save_dir == "none":
        args.save_dir = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.video:
        gen = frames_from_video(args.video, args.size)
    elif args.frames_dir:
        gen = frames_from_dir(args.frames_dir, args.size)
    else:
        clip, _, _ = next(iter(SynthClips(args.size, clip_len=args.frames)))
        gen = iter(clip)

    state, prev_depth, ratios = None, None, []
    with torch.no_grad():
        for i, frame in enumerate(gen):
            if i >= args.frames:
                break
            depth, state, info = model.step(frame.unsqueeze(0).to(dev), state)
            ratios.append(info["active_ratio"])
            flick = ((depth - prev_depth).abs().mean().item()
                     if prev_depth is not None else 0.0)
            prev_depth = depth
            if args.save_dir:
                save_depth(depth, os.path.join(args.save_dir, f"depth_{i:06d}.png"))
            print(f"frame {i:3d}  active {info['active_ratio']*100:5.1f}%  "
                  f"depth-delta {flick:.4f}")

    avg = sum(ratios) / len(ratios)
    print(f"\navg active ratio {avg*100:.1f}%  ->  "
          f"temporal-backbone compute ~{avg*100:.1f}% of full")


if __name__ == "__main__":
    main()
