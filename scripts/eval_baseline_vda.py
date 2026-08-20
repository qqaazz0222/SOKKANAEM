"""Video Depth Anything (metric) baseline on the exact same holdout clips/
protocol as scripts/eval.py, for a fair comparison against SOKKANAEM.

The one true *video* baseline here — unlike DA v2 (per-frame) and DA3
(any-view, jointly-processed-but-not-causal), VDA has an explicit temporal
mechanism (motion_module) run causally over the clip via infer_video_depth().

Runs in a separate `vda` conda env (their requirements.txt hard-pins
torch==2.1.1 — genuinely incompatible with the `sokkanaem`/`baselines`
envs' torch 2.8, hence the third env):
    conda create -n vda python=3.10
    conda run -n vda pip install torch==2.1.1 torchvision==0.16.1
    conda run -n vda pip install numpy==1.24.0 opencv-python matplotlib \
        pillow imageio==2.37.0 einops easydict tqdm huggingface_hub
    conda run -n vda pip install -e /workspace/SOKKANAEM --no-deps
    # xformers skipped — their attention/motion_module fall back to plain
    # PyTorch when it's absent (XFORMERS_AVAILABLE guard), same as DA3.
    git clone https://github.com/DepthAnything/Video-Depth-Anything <VDA_DIR>
    # download depth-anything/Metric-Video-Depth-Anything-Small ->
    # <VDA_DIR>/checkpoints/metric_video_depth_anything_vits.pth
    conda run -n vda python scripts/eval_baseline_vda.py <VDA_DIR>

Metric checkpoint output correlates with gt_depth at 0.80 but the median
ratio isn't 1:1 (0.74) — their metric calibration doesn't transfer exactly
to this synthetic domain's camera intrinsics. Same median-scale-only
alignment as eval_once() (not scale+shift — this model claims metric
output, so scale-only is the fairer, less generous protocol, matching
exactly how SOKKANAEM itself is scored).
"""
import os
import sys

import numpy as np
import torch

from sokkanaem.data import (build_mixed, eval_clip_len,
                            eval_set_from_env)
from sokkanaem.metrics import clip_scores, report


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/eval_baseline_vda.py <Video-Depth-Anything checkout dir>")
        sys.exit(1)
    sys.path.insert(0, sys.argv[1])
    from video_depth_anything.video_depth import VideoDepthAnything

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoDepthAnything(encoder="vits", features=64,
                               out_channels=[48, 96, 192, 384], metric=True)
    model.load_state_dict(
        torch.load(f"{sys.argv[1]}/checkpoints/metric_video_depth_anything_vits.pth",
                   map_location="cpu"),
        strict=True)
    model = model.to(dev).eval()

    specs, holdout, tag = eval_set_from_env(
        ["vkitti2:/home/hyunsu/dataset_ssd/vkitti2",
         "tartanair2:/home/hyunsu/dataset_ssd/tartanair_v2",
         "pointodyssey:/home/hyunsu/dataset_ssd/pointodyssey"],
        ["Scene06", "OldTownFall", "/pointodyssey/val/", "/pointodyssey/test/"])
    dataset, _ = build_mixed(
        specs, clip_len=(_cl := eval_clip_len())[0], clip_stride=_cl[1],
        size=256, holdout=holdout, val=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 1000, not 100: 1.1% of the holdout was not a representative sample
    # (SOKKANAEM's own delta1 moved 0.397 -> 0.544 on the same protocol).
    max_clips = int(os.environ.get("MAX_CLIPS", 1000))
    acc = {}
    for ci, (clip, gt, valid) in enumerate(loader):
        if ci >= max_clips:
            break
        frames_np = (clip[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)  # (T,H,W,3)
        depths, _ = model.infer_video_depth(frames_np, target_fps=8, input_size=256,
                                            device=dev, fp32=True)
        d = torch.from_numpy(depths)  # (T,H,W)

        v = valid[0, :, 0].bool()
        gtd = gt[0, :, 0]
        s = gtd[v].median() / d[v].median().clamp(min=1e-6)
        depth_pred = d * s

        # shared scorer (sokkanaem/metrics.py) — same protocol + OPW/TCE
        sc = clip_scores(clip[0], depth_pred.unsqueeze(1),
                         gtd.unsqueeze(1), valid[0])
        if sc is None:   # clip with no valid GT pixel — see clip_scores
            continue
        for k, val in sc.items():
            acc.setdefault(k, []).append(val)
        if (ci + 1) % 100 == 0:
            print(f"  {ci+1}/{max_clips} clips...", file=sys.stderr)

    report(f"Video-Depth-Anything-Small metric (~28.4M){tag}", acc)


if __name__ == "__main__":
    main()
