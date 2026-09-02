"""Q0: train the metric calibration head on top of a frozen DA2 shape branch.

Everything except two numbers per frame is frozen, so this is a small,
fast run whose only question is the one PLAN §7 asks: with a shape model that
already passes the sharpness gate, does OUR metric protocol reach the accuracy
gate? A failure here is decisive -- it would mean the gate is not about shape
quality at all.

    python scripts/train_q.py --steps 4000 --work-dir work_dirs/q0-calib-s0
"""
import argparse
import logging
import random
import sys
import tomllib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import write_config                    # same provenance record
from sokkanaem.data import build_mixed
from sokkanaem.distill import affine_invariant_loss
from sokkanaem.losses import (boundary_location_loss, si_log_loss,
                              warp_residual_loss)
from sokkanaem.qmodel import QDepth
from sokkanaem.schedule import lr_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main_v8.toml",
                    help="only its data/holdout/size are used")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--clip-len", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--infer-size", type=int, default=518,
                    help="DA2's own input resolution; the published quality is "
                         "a property of it")
    ap.add_argument("--real-only", action="store_true",
                    help="metric scale is what is being learned and the two "
                         "real sources are the ones the gate is measured on")
    ap.add_argument("--unfreeze", action="store_true",
                    help="Q1: train the DA2 shape branch too, at --shape-lr. "
                         "Q0's failure mode is inherited, not learned -- 10 of "
                         "its clips fail the 2-DOF fit because DA2's TUM shape "
                         "does, and a frozen branch cannot fix that")
    ap.add_argument("--temporal", action="store_true",
                    help="Stage E's first step: a zero-init temporal residual "
                         "adapter between DA2's backbone and neck. A fresh one "
                         "reproduces Q0 to 1e-5, so the sealed floor cannot be "
                         "lost by adding streaming")
    ap.add_argument("--warp-weight", type=float, default=0.0,
                    help="training-time TCE (the adapter's actual job -- state "
                         "does not create per-frame accuracy, REPORT 4.43)")
    ap.add_argument("--unfreeze-head", action="store_true",
                    help="Q1b: train the DPT neck+head only, leaving the "
                         "DINOv2 encoder frozen. The encoder is what 142M "
                         "images bought; 13k indoor clips are not an argument "
                         "for moving it, and the fully unfrozen arm regressed "
                         "on every gauge (REPORT 4.45)")
    ap.add_argument("--shape-lr", type=float, default=1e-5,
                    help="LR for the pretrained branch; the head keeps --lr")
    ap.add_argument("--gt-weight", type=float, default=1.0,
                    help="weight on the si-log term against the sensor GT. The "
                         "temporal adapter arm turns this DOWN: trained on our "
                         "GT it dragged DA2's shape down exactly as Q1/Q2/S1 "
                         "did (grad_ratio 0.709 -> 0.409), and its job is "
                         "stability, not shape")
    ap.add_argument("--retention-weight", type=float, default=1.0,
                    help="scale-shift-invariant match to the FROZEN DA2 "
                         "disparity, so unfreezing cannot quietly trade the "
                         "sharpness this track exists to keep")
    ap.add_argument("--boundary-weight", type=float, default=0.0)
    ap.add_argument("--work-dir", default="work_dirs/q0-calib-s0")
    args = ap.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    data = cfg["data"][:2] if args.real_only else cfg["data"]
    holdout = cfg["holdout"]

    train_shape = args.unfreeze or args.unfreeze_head
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    model_kw = {"kind": "q", "infer_size": args.infer_size,
                "freeze_shape": not train_shape, "temporal": args.temporal}
    meta = write_config(work / "config.toml", args, model_kw)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(work / "train.log")])
    log = logging.getLogger("train_q").info
    log(f"work dir: {work}  commit {meta['git_commit'][:8]}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = QDepth(infer_size=args.infer_size, temporal=args.temporal,
                   freeze_shape=not train_shape).to(dev)
    if args.unfreeze_head:
        model.net.backbone.eval()
        for prm in model.net.backbone.parameters():
            prm.requires_grad_(False)
    teacher = None
    if (train_shape or args.temporal) and args.retention_weight > 0:
        # a second, frozen copy: the retention target has to be the ORIGINAL
        # DA2, not the branch being updated
        teacher = QDepth(infer_size=args.infer_size).to(dev).eval()
    groups = [{"params": list(model.calib.parameters()), "lr": args.lr}]
    if args.temporal:
        groups.append({"params": list(model.temporal.parameters()),
                       "lr": args.lr})
    if train_shape:
        (model.net.neck if args.unfreeze_head else model.net).train()
        tuned = ([*model.net.neck.parameters(), *model.net.head.parameters()]
                 if args.unfreeze_head else list(model.net.parameters()))
        groups.append({"params": [p for p in tuned if p.requires_grad],
                       "lr": args.shape_lr})
    else:
        model.net.eval()                              # frozen, and stays eval
    params = [p for g in groups for p in g["params"] if p.requires_grad]
    log(f"trainable {sum(p.numel() for p in params)/1e6:.3f}M of "
        f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M"
        + (f"  (shape lr {args.shape_lr}"
           f"{', head only' if args.unfreeze_head else ''})"
           if train_shape else ""))
    opt = torch.optim.AdamW(groups, lr=args.lr)
    base_lrs = [g["lr"] for g in opt.param_groups]

    dataset, sampler = build_mixed(data, clip_len=args.clip_len, size=args.size,
                                   holdout=holdout, augment=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, sampler=sampler,
        num_workers=args.workers, drop_last=True, pin_memory=True,
        persistent_workers=args.workers > 0)
    log(f"mixed dataset: {len(dataset)} clips from {len(data)} sources")

    step = 0
    while step < args.steps:
        for clip, gt, valid in loader:
            if step >= args.steps:
                break
            clip, gt, valid = clip.to(dev), gt.to(dev), valid.to(dev)
            depths, _ = model.forward_clip(clip)
            loss = args.gt_weight * si_log_loss(depths, gt, valid)
            if teacher is not None:
                B, T = clip.shape[:2]
                flat = clip.reshape(B * T, *clip.shape[-3:])
                with torch.no_grad():
                    ref, _ = teacher._shape(flat)
                loss = loss + args.retention_weight * affine_invariant_loss(
                    depths.reshape(B * T, 1, *depths.shape[-2:]), ref,
                    valid.reshape(B * T, 1, *valid.shape[-2:]))
            if args.warp_weight > 0:
                loss = loss + args.warp_weight * warp_residual_loss(
                    clip, depths, gt, valid)
            if args.boundary_weight > 0:
                loss = loss + args.boundary_weight * boundary_location_loss(
                    depths, gt, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = lr_at(step, base, args.warmup, args.steps)
            opt.step()
            if step % 50 == 0:
                log(f"step {step:6d}  loss {loss.item():.4f}  "
                    f"median {depths.median().item():.2f} m")
            step += 1
    torch.save({"meta": {**meta, "args": vars(args), "model": model_kw},
                "model": model.state_dict(), "step": step},
               work / "latest.pt")
    log(f"saved -> {work}/latest.pt")


if __name__ == "__main__":
    main()
