"""Depth Anything 3 baseline on the exact same holdout clips/protocol as
scripts/eval.py, for a fair comparison against SOKKANAEM.

Runs in the separate `baselines` conda env (torch 2.8.0+cu128, no xformers —
DA3's SwiGLU layer falls back to pure PyTorch when xformers is absent):
    conda env create -n baselines python=3.11
    conda run -n baselines pip install torch==2.8.0 torchvision numpy<2 pillow \
        einops huggingface_hub imageio opencv-python safetensors omegaconf \
        trimesh open3d fastapi uvicorn requests typer evo e3nn \
        moviepy==1.0.3 plyfile pillow_heif pycolmap
    conda run -n baselines pip install -e <depth-anything-3 checkout> --no-deps
    conda run -n baselines pip install -e /workspace/SOKKANAEM --no-deps
    conda run -n baselines python scripts/eval_baseline_da3.py

DA3 is an any-view (multi-image, not causal-video) model — fed the whole
8-frame clip jointly via model.inference(), giving it whatever cross-frame
consistency its architecture provides "for free" (unlike DA v2's per-frame
independence). Output correlates positively with gt_depth (0.886, vs -0.79
for 1/gt_depth) — it's depth-like, not disparity, so alignment is a direct
scale+shift fit in depth space (no inversion needed).
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from depth_anything_3.api import DepthAnything3

from sokkanaem.data import build_mixed, eval_set_from_env
from sokkanaem.metrics import clip_scores, report

CKPT = "depth-anything/DA3-BASE"


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(CKPT).to(device=dev)

    specs, holdout, tag = eval_set_from_env(
        ["vkitti2:/home/hyunsu/dataset_ssd/vkitti2",
         "tartanair2:/home/hyunsu/dataset_ssd/tartanair_v2",
         "pointodyssey:/home/hyunsu/dataset_ssd/pointodyssey"],
        ["Scene06", "OldTownFall", "/pointodyssey/val/", "/pointodyssey/test/"])
    dataset, _ = build_mixed(
        specs, clip_len=8, clip_stride=8, size=256, holdout=holdout, val=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 1000, not 100: on the identical protocol SOKKANAEM's own delta1 moved
    # 0.397 -> 0.544 between 100 and 1000 clips (1.1% of the holdout was not
    # a representative sample). Deterministic first-N, same set per model.
    max_clips = int(os.environ.get("MAX_CLIPS", 1000))
    acc = {}
    for ci, (clip, gt, valid) in enumerate(loader):
        if ci >= max_clips:
            break
        T = clip.shape[1]
        imgs = [(clip[0, t].permute(1, 2, 0).numpy() * 255).astype("uint8") for t in range(T)]
        pred = model.inference(imgs)
        d = torch.from_numpy(pred.depth).float().unsqueeze(1)  # (T,1,h,w)
        d = F.interpolate(d, size=clip.shape[-2:], mode="bilinear",
                          align_corners=False).squeeze(1)  # (T,H,W)

        v = valid[0, :, 0].bool()
        gtd = gt[0, :, 0]
        x = d[v].numpy().astype(np.float64)
        y = gtd[v].numpy().astype(np.float64)
        A = np.stack([x, np.ones_like(x)], axis=1)
        (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        depth_pred = torch.from_numpy(s * d.numpy() + b).float().clamp(min=1e-3)

        # shared scorer (sokkanaem/metrics.py) — identical protocol for every
        # model, including the OPW/TCE temporal metrics
        sc = clip_scores(clip[0], depth_pred.unsqueeze(1),
                         gtd.unsqueeze(1), valid[0])
        if sc is None:   # clip with no valid GT pixel — see clip_scores
            continue
        for k, val in sc.items():
            acc.setdefault(k, []).append(val)
        if (ci + 1) % 100 == 0:
            print(f"  {ci+1}/{max_clips} clips...", file=sys.stderr)

    report(f"DA3-BASE (0.12B){tag}", acc)


if __name__ == "__main__":
    main()
