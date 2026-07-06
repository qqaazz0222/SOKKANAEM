"""Dataset preparation and validation.

check       validate --data specs before training: sequences, clips,
            depth range / valid ratio (catches wrong scale or layout early)
from-video  extract video into the generic folder layout (out/seq0/rgb/*.png)
            for inference; no depth GT is produced

Run:
    python scripts/prepare_data.py check --data scannet:/data/scannet --data kitti:/data/kitti
    python scripts/prepare_data.py from-video cam.mp4 --out data/cctv
"""
import argparse
import os

from sokkanaem.data import ADAPTERS, ClipDataset


def cmd_check(args):
    for spec in args.data:
        parts = spec.split(":")
        name, root = parts[0], parts[1]
        fn, scale = ADAPTERS[name]
        if len(parts) > 2:
            scale = float(parts[2])
        seqs = fn(root)
        n_frames = sum(len(s) for s in seqs)
        ds = ClipDataset(seqs, scale, clip_len=args.clip_len, size=args.size)
        print(f"\n[{spec}]")
        print(f"  sequences {len(seqs)}  frames {n_frames}  "
              f"clips(T={args.clip_len}) {len(ds)}")
        if len(ds) == 0:
            print("  !! no clips — check layout (expected: see sokkanaem/data.py)")
            continue
        _, depth, valid = ds[0]
        d = depth[valid.bool()]
        print(f"  depth[m] min {d.min():.2f}  median {d.median():.2f}  "
              f"max {d.max():.2f}  valid {valid.mean()*100:.0f}%")
        if d.median() > 100 or d.median() < 0.05:
            print(f"  !! implausible median depth — wrong scale? (using {scale})")


def cmd_from_video(args):
    import cv2
    out = os.path.join(args.out, "seq0", "rgb")
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    i = saved = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % args.every == 0:
            cv2.imwrite(os.path.join(out, f"{saved:06d}.png"), img)
            saved += 1
        i += 1
    print(f"{saved} frames -> {out}")
    print(f"inference: python scripts/infer.py --frames-dir {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="validate dataset specs")
    c.add_argument("--data", action="append", required=True)
    c.add_argument("--clip-len", type=int, default=4)
    c.add_argument("--size", type=int, default=128)
    c.set_defaults(fn=cmd_check)

    v = sub.add_parser("from-video", help="video -> folder layout (rgb only)")
    v.add_argument("video")
    v.add_argument("--out", required=True)
    v.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    v.set_defaults(fn=cmd_from_video)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
