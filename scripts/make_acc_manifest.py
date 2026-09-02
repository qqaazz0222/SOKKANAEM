"""Seal the ACC comparison's clip set: PLAN_ACC A1.

Writes manifests/acc_real_L{N}.json naming every frame of every clip, so the
G1/G2 tables cannot be assembled from two different sets of them. `build_mixed`
+ `even_subset` reproduce a clip set only while the adapters, holdout strings,
glob order and per-source cap all hold still; three of those have already
changed once in this repo's history (REPORT §4.10, and the first-N truncation
that made "Bonn" mean rgbd_bonn_crowd2 alone).

Every clip in the holdout is taken -- no cap. The cap existed to bound baseline
inference cost, and it is exactly the thing A1 says the comparison must not
depend on. TUM+Bonn at L8 is 488 clips, which is minutes of DPT-Large.

The valid-pixel count of each clip is recorded because G1 disqualifies on
per-clip failures, and a clip that is 2% valid produces a per-clip AbsRel with
no business carrying the same weight as a full one. It also pins the depth
loader: a manifest whose valid counts no longer reproduce means the decode path
changed underneath the table.

    python scripts/make_acc_manifest.py --clip-len 8
    python scripts/make_acc_manifest.py --clip-len 256 --out manifests/x.json
"""
import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import torch

from sokkanaem.data import ADAPTERS, build_mixed

D = "/home/hyunsu/dataset_ssd"
SPECS = [f"tum:{D}/tum_static", f"bonn:{D}/bonn/rgbd_bonn_dataset"]
# the holdout of configs/main_v8.toml onward, unchanged -- these sequences have
# never been trained on and every reported number already uses them
HOLDOUT = ["walking_static", "rgbd_bonn_crowd2", "rgbd_bonn_person_tracking2",
           "rgbd_bonn_static_close_far"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default=None)
    ap.add_argument("--spec", action="append", default=None)
    ap.add_argument("--holdout", action="append", default=None)
    args = ap.parse_args()

    specs = args.spec or SPECS
    holdout = args.holdout or HOLDOUT
    out = Path(args.out or f"manifests/acc_real_L{args.clip_len}.json")

    dataset, _ = build_mixed(specs, clip_len=args.clip_len,
                             clip_stride=args.clip_len, size=args.size,
                             holdout=holdout, val=True)

    sources, clips = {}, []
    for spec, ds in zip(specs, dataset.datasets):
        name = spec.split(":")[0]
        sources[name] = {"spec": spec, "scale": ds.scale, "mode": ds.mode}
        for i in range(len(ds)):
            seq, start, fs = ds.clips[i]
            pairs = seq[start:start + args.clip_len * fs:fs]
            _, _, valid = ds[i]
            clips.append({
                "source": name,
                "pairs": [list(p) for p in pairs],
                "valid_px": int(valid.sum().item()),
                "px": int(valid.numel()),
            })
        print(f"{name}: {len(ds)} clips")

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": 1, "created": str(date.today()), "git": git,
        "clip_len": args.clip_len, "frame_stride": 1, "size": args.size,
        "specs": specs, "holdout": holdout,
        "sources": sources, "clips": clips,
    }, indent=1))
    valid = sum(c["valid_px"] for c in clips) / max(
        sum(c["px"] for c in clips), 1)
    print(f"-> {out}  {len(clips)} clips, {valid*100:.1f}% valid pixels")

    # the manifest is only worth anything if it reloads to the same tensors
    from sokkanaem.data import load_manifest
    for src, ds in load_manifest(out):
        ref = [c for c in clips if c["source"] == src]
        assert len(ds) == len(ref), (src, len(ds), len(ref))
        _, _, v = ds[0]
        assert int(v.sum().item()) == ref[0]["valid_px"], src
    print("reload check ok")


if __name__ == "__main__":
    main()
