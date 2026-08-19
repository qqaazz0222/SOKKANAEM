"""Does accuracy improve as the temporal state warms up?

A streaming model carries evidence across frames, so depth at frame 7 should
be better than at frame 0 -- that is the whole reason to keep state. Nobody
had measured it. If the curve is flat, the temporal state is buying stability
only, and accuracy has to come from somewhere else.
"""
import sys, torch
sys.path.insert(0, "/workspace/SOKKANAEM")
from sokkanaem import from_checkpoint
from sokkanaem.data import build_mixed

import os
CLIP_LEN = int(os.environ.get("CLIP_LEN", 32))
MAX_CLIPS = int(os.environ.get("MAX_CLIPS", 30))
dev = "cuda"
m = from_checkpoint("work_dirs/v9-60k/latest.pt", dev).eval()
D = "/home/hyunsu/dataset_ssd"
for name, spec, hold in (("tum", f"tum:{D}/tum_static", ["walking_static"]),
                         ("bonn", f"bonn:{D}/bonn/rgbd_bonn_dataset",
                          ["rgbd_bonn_crowd2","rgbd_bonn_person_tracking2","rgbd_bonn_static_close_far"])):
    ds, _ = build_mixed([spec], clip_len=CLIP_LEN, clip_stride=CLIP_LEN, size=256, holdout=hold, val=True)
    ld = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    per_t = [[] for _ in range(CLIP_LEN)]
    with torch.no_grad():
        for ci, (clip, gt, valid) in enumerate(ld):
            if ci >= MAX_CLIPS: break
            clip, gt, valid = clip.to(dev), gt.to(dev), valid.to(dev)
            depths, _ = m.forward_clip(clip)
            for t in range(clip.shape[1]):
                v = valid[0, t].bool()
                if not v.any(): continue
                # align each FRAME independently so the curve is not an
                # artefact of one clip-level scale fitted mostly to late frames
                s = gt[0, t][v].median() / depths[0, t][v].median().clamp(min=1e-6)
                p = depths[0, t] * s
                per_t[t].append(((p[v]-gt[0,t][v]).abs()/gt[0,t][v].clamp(min=1e-6)).mean().item())
    print(f"\n{name}: AbsRel by frame index (state warm-up)")
    for t, xs in enumerate(per_t):
        if xs: print(f"  frame {t:2d}: {sum(xs)/len(xs):.4f}")
