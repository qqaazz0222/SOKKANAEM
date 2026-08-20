"""Hugging Face depth baselines on the exact same holdout clips/protocol as
scripts/eval.py, for a fair comparison against SOKKANAEM.

Defaults to Depth Anything V2 Small; --ckpt runs any other HF depth model so
the commonly cited comparison group (DPT/MiDaS, the Depth Anything sizes, a
metric model) can be measured under one protocol instead of quoted from
papers that each use their own.

Requires: pip install -e ".[baseline]"

DA v2 is a relative (disparity-like) single-frame model — confirmed via
correlation with 1/gt_depth (0.988) vs gt_depth (-0.64) on a real holdout
frame. To match our own eval's per-CLIP (not per-frame) scale alignment,
fit one scale+shift in disparity space per clip (all 8 frames' valid
pixels combined, standard scale-shift-invariant protocol), then invert to
depth before computing AbsRel/RMSE/delta1/t-delta the same way eval_once does.

Run:
    python scripts/eval_baseline_da2.py
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from sokkanaem.data import (build_mixed, eval_clip_len,
                            eval_set_from_env)
from sokkanaem.metrics import clip_scores, report

CKPT = "depth-anything/Depth-Anything-V2-Small-hf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT, help="HF depth model id")
    ap.add_argument("--label", default=None, help="row name in the report")
    ap.add_argument("--mode", default="relative",
                    choices=["relative", "metric"],
                    help="relative: least-squares scale+shift in disparity "
                         "space (MiDaS protocol). metric: per-clip median "
                         "scaling, the same alignment scripts/eval.py uses.")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(args.ckpt)
    model = AutoModelForDepthEstimation.from_pretrained(args.ckpt).to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6

    specs, holdout, tag = eval_set_from_env(
        ["vkitti2:/home/hyunsu/dataset_ssd/vkitti2",
         "tartanair2:/home/hyunsu/dataset_ssd/tartanair_v2",
         "pointodyssey:/home/hyunsu/dataset_ssd/pointodyssey"],
        ["Scene06", "OldTownFall", "/pointodyssey/val/", "/pointodyssey/test/"])
    dataset, _ = build_mixed(
        specs, clip_len=(_cl := eval_clip_len())[0], clip_stride=_cl[1],
        size=256, holdout=holdout, val=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 100 clips was 1.1% of the holdout and wildly unrepresentative — on the
    # identical protocol SOKKANAEM's own delta1 moved 0.397 -> 0.544 going
    # from 100 to 1000 clips. Deterministic first-N, same set for every model.
    max_clips = int(os.environ.get("MAX_CLIPS", 1000))
    acc = {}
    with torch.no_grad():
        for ci, (clip, gt, valid) in enumerate(loader):
            if ci >= max_clips:
                break
            T = clip.shape[1]
            preds = []
            for t in range(T):
                frame = clip[0, t]  # (3,H,W) in [0,1]
                img = Image.fromarray((frame.permute(1, 2, 0).numpy() * 255).astype("uint8"))
                inputs = proc(images=img, return_tensors="pt").to(dev)
                out = model(**inputs)
                # ZoeDepth's processor pads before inference and needs to
                # be told the pre-pad size to undo it; the others do not take
                # the argument at all.
                pp = dict(target_sizes=[clip.shape[-2:]])
                if type(proc).__name__.startswith("ZoeDepth"):
                    pp["source_sizes"] = [clip.shape[-2:]]
                post = proc.post_process_depth_estimation(out, **pp)
                preds.append(post[0]["predicted_depth"].cpu())
            pred = torch.stack(preds)  # (T,H,W) disparity-like

            v = valid[0, :, 0].bool()  # (T,H,W)
            gtd = gt[0, :, 0]          # (T,H,W) metric depth
            if args.mode == "relative":
                disp_gt = 1.0 / gtd.clamp(min=1e-3)
                x = pred[v].numpy().astype(np.float64)
                y = disp_gt[v].numpy().astype(np.float64)
                # least-squares scale+shift: y ~= s*x + b (standard scale-
                # shift-invariant protocol for relative-depth models)
                A = np.stack([x, np.ones_like(x)], axis=1)
                (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
                aligned_disp = s * pred.numpy() + b
                depth_pred = torch.from_numpy(
                    1.0 / np.clip(aligned_disp, 1e-3, None)).float()
            else:
                # metric model: align exactly the way scripts/eval.py does
                scale = gtd[v].median() / pred[v].median().clamp(min=1e-6)
                depth_pred = pred * scale

            # one shared scorer (sokkanaem/metrics.py) for every model —
            # this file and eval.py had already drifted on t-delta once
            sc = clip_scores(clip[0], depth_pred.unsqueeze(1),
                             gtd.unsqueeze(1), valid[0])
            if sc is None:   # clip with no valid GT pixel — see clip_scores
                continue
            for k, val in sc.items():
                acc.setdefault(k, []).append(val)
            if (ci + 1) % 100 == 0:
                print(f"  {ci+1}/{max_clips} clips...", file=sys.stderr)

    label = args.label or args.ckpt.split("/")[-1]
    report(f"{label} ({n_par:.1f}M){tag}", acc)


if __name__ == "__main__":
    main()
