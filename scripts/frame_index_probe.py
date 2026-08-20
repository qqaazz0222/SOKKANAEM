"""Does accuracy improve as the temporal state warms up?

A streaming model carries evidence across frames, so depth at frame 7 should
be better than at frame 0 -- that is the whole reason to keep state. If the
curve is flat, the temporal state is buying stability only, and accuracy has
to come from somewhere else.

AbsRel alone cannot say whether the stability lead survives a long stream, so
delta1, raw frame difference, OPW and TCE are scored by frame index too. Each
frame is aligned independently, so the curve is not an artefact of one
clip-level scale fitted mostly to late frames.

    python scripts/frame_index_probe.py --ckpt work_dirs/v9-60k/latest.pt
"""
import argparse
import sys

import torch

sys.path.insert(0, "/workspace/SOKKANAEM")
from sokkanaem import from_checkpoint
from sokkanaem.data import build_mixed
from sokkanaem.metrics import temporal_metrics

D = "/home/hyunsu/dataset_ssd"
SOURCES = (("tum", f"tum:{D}/tum_static", ["walking_static"]),
           ("bonn", f"bonn:{D}/bonn/rgbd_bonn_dataset",
            ["rgbd_bonn_crowd2", "rgbd_bonn_person_tracking2",
             "rgbd_bonn_static_close_far"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/v9-60k/latest.pt")
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--clip-stride", type=int, default=None)
    ap.add_argument("--max-clips", type=int, default=30)
    ap.add_argument("--keyframe-every", type=int, default=None)
    ap.add_argument("--every", type=int, default=1,
                    help="print every Nth frame index (a 256-frame curve does "
                         "not need 256 rows)")
    args = ap.parse_args()
    dev = "cuda"
    m = from_checkpoint(args.ckpt, dev).eval()
    if args.keyframe_every:
        m.detector.keyframe_every = args.keyframe_every
    print(f"ckpt={args.ckpt} clip_len={args.clip_len} "
          f"stride={args.clip_stride or args.clip_len} max={args.max_clips} "
          f"keyframe_every={m.detector.keyframe_every}")

    for name, spec, hold in SOURCES:
        ds, _ = build_mixed([spec], clip_len=args.clip_len,
                            clip_stride=args.clip_stride or args.clip_len,
                            size=256, holdout=hold, val=True)
        ld = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
        keys = ("absrel", "delta1", "tdelta", "opw", "tce")
        per_t = {k: [[] for _ in range(args.clip_len)] for k in keys}
        with torch.no_grad():
            for ci, (clip, gt, valid) in enumerate(ld):
                if ci >= args.max_clips:
                    break
                clip, gt, valid = clip.to(dev), gt.to(dev), valid.to(dev)
                depths, _ = m.forward_clip(clip)
                v = valid[0].bool()
                # per-frame median alignment: the curve must not inherit one
                # clip-level scale fitted mostly to the late frames
                p = torch.zeros_like(depths[0])
                for t in range(clip.shape[1]):
                    if not v[t].any():
                        continue
                    s = gt[0, t][v[t]].median() / depths[0, t][v[t]].median().clamp(min=1e-6)
                    p[t] = depths[0, t] * s
                    err = (p[t][v[t]] - gt[0, t][v[t]]).abs() / gt[0, t][v[t]].clamp(min=1e-6)
                    per_t["absrel"][t].append(err.mean().item())
                    r = torch.maximum(p[t][v[t]] / gt[0, t][v[t]].clamp(min=1e-6),
                                      gt[0, t][v[t]] / p[t][v[t]].clamp(min=1e-6))
                    per_t["delta1"][t].append((r < 1.25).float().mean().item())
                    if t > 0:
                        per_t["tdelta"][t].append(
                            (p[t] - p[t - 1]).abs().mean().item())
                        # OPW/TCE are pairwise, so score frame t on the pair
                        # (t-1, t) -- one RAFT call per index, same total flow
                        tm = temporal_metrics(clip[0, t - 1:t + 1],
                                              p[t - 1:t + 1], gt[0, t - 1:t + 1],
                                              valid[0, t - 1:t + 1])
                        per_t["opw"][t].append(tm["opw"])
                        per_t["tce"][t].append(tm["tce"])
        print(f"\n{name}: by frame index ({len(ds)} clips available, "
              f"{min(args.max_clips, len(ds))} scored)")
        print("  frame  AbsRel  delta1  t-delta     OPW     TCE")
        for t in range(args.clip_len):
            if t % args.every and t != args.clip_len - 1:
                continue
            xs = per_t["absrel"][t]
            if not xs:
                continue
            cell = lambda k: (f"{sum(per_t[k][t]) / len(per_t[k][t]):.4f}"
                              if per_t[k][t] else "     -")
            print(f"  {t:5d}  {cell('absrel')}  {cell('delta1')}  "
                  f"{cell('tdelta')}  {cell('opw')}  {cell('tce')}")


if __name__ == "__main__":
    main()
