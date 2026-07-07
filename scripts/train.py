"""Training (IDEA.md §3.3–3.4).

Real datasets via --data (spec strings in sokkanaem/data.py), synthetic
moving-box fallback when omitted. Random-mask-ratio scheduling for
sparsity robustness.

Run:
    python scripts/train.py --steps 500                    # synthetic
    python scripts/train.py --config configs/scannet.toml --data scannet:/data/scannet
    python scripts/train.py --data scannet:/data/scannet --data kitti:/data/kitti

Config: TOML file sets defaults; explicit CLI flags override. [model]
table is passed to SOKKANAEM(**model) — detector thresholds live there.

Outputs (checkpoint latest.pt, train.log, config.toml copy) go to
work_dirs/<config name>/ (override with --work-dir).
"""
import argparse
import logging
import shutil
import tomllib
from pathlib import Path

import torch

from sokkanaem import SOKKANAEM
from sokkanaem.data import SynthClips, build_mixed
from sokkanaem.losses import grad_loss, si_log_loss, temporal_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="TOML config path")
    ap.add_argument("--data", action="append", default=None,
                    help="dataset spec 'name:/root[:scale]', repeatable; "
                         "omit for synthetic data")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--clip-len", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-skip", type=float, default=0.8,
                    help="final random-mask skip ratio")
    ap.add_argument("--detector-mask", action="store_true",
                    help="train with detector-driven masks instead of iid "
                         "random (§4.4 mask-distribution ablation)")
    ap.add_argument("--holdout", action="append", default=None,
                    help="path substring for the val split (repeatable); "
                         "matching sequences are excluded from training")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--work-dir", default=None,
                    help="output dir; default work_dirs/<config name>")

    # config sets defaults, explicit CLI flags win
    pre, _ = ap.parse_known_args()
    model_kw = {}
    if pre.config:
        with open(pre.config, "rb") as f:
            cfg = tomllib.load(f)
        model_kw = cfg.pop("model", {})
        # data roots stay on CLI; config may hold dataset names to prefix
        specs = cfg.pop("data", None)
        if specs and pre.data is None:
            ap.set_defaults(data=specs)
        ap.set_defaults(**cfg)
    args = ap.parse_args()

    name = Path(args.config).stem if args.config else "default"
    work = Path(args.work_dir or f"work_dirs/{name}")
    work.mkdir(parents=True, exist_ok=True)
    if args.config:
        shutil.copy(args.config, work / "config.toml")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(work / "train.log")])
    log = logging.getLogger("train").info
    log(f"work dir: {work}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SOKKANAEM(**model_kw).to(dev)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.data:
        dataset, sampler = build_mixed(args.data, clip_len=args.clip_len,
                                       size=args.size, holdout=args.holdout)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch, sampler=sampler,
            num_workers=args.workers, drop_last=True)
        log(f"mixed dataset: {len(dataset)} clips from {len(args.data)} sources")
    else:
        loader = torch.utils.data.DataLoader(
            SynthClips(args.size, args.clip_len), batch_size=args.batch)

    N = (args.size // 16) ** 2
    step = 0
    while step < args.steps:
        for clip, gt, valid in loader:
            if step >= args.steps:
                break
            clip, gt, valid = clip.to(dev), gt.to(dev), valid.to(dev)
            B, T = clip.shape[:2]

            if args.detector_mask:
                skip = 0.0
                depths, masks = model.forward_clip(clip)  # detector-driven
            else:
                # random-mask scheduling: skip ratio ramps 0 -> max (§3.2)
                skip = args.max_skip * min(1.0, step / max(1, args.steps // 2))
                fm = (torch.rand(B, T, N, device=dev) > skip).float()
                fm[:, 0] = 1.0  # first frame always full
                depths, masks = model.forward_clip(clip, force_mask=fm)
            loss = (si_log_loss(depths, gt, valid)
                    + 0.5 * grad_loss(depths, gt, valid)
                    + 0.1 * temporal_loss(depths, masks))
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 50 == 0:
                log(f"step {step:4d}  loss {loss.item():.4f}  skip {skip:.2f}")
            step += 1
            if step % 2000 == 0:  # crash insurance for long runs
                torch.save(model.state_dict(), work / "latest.pt")

    ckpt = work / "latest.pt"
    torch.save(model.state_dict(), ckpt)
    log(f"saved -> {ckpt}")


if __name__ == "__main__":
    main()
