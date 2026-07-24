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
from sokkanaem.ema import ema_update_
from sokkanaem.losses import grad_loss, normal_loss, si_log_loss, temporal_loss
from sokkanaem.schedule import lr_at, parse_size_schedule, size_for_step


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
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="eval-time shadow-weight EMA decay (0 disables)")
    ap.add_argument("--auto-loss-weight", action="store_true",
                    help="Kendall multi-task uncertainty weighting instead "
                         "of the fixed si_log/grad/temporal/normal weights")
    ap.add_argument("--normal-weight", type=float, default=0.0,
                    help="surface-normal loss weight (ignored if "
                         "--auto-loss-weight; 0 = off, matches old default)")
    ap.add_argument("--size-schedule", default=None,
                    help="progressive resolution curriculum: "
                         "'step:size,step:size,...' e.g. '0:128,20000:256' "
                         "(default: fixed --size throughout)")
    ap.add_argument("--warmup", type=int, default=2000,
                    help="LR linear-warmup steps before cosine decay "
                         "(--warmup 0 disables the whole schedule -> flat lr)")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="max grad norm (0 disables)")

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
    # Kendall multi-task uncertainty weighting (§ auto-loss-weight): learnable
    # log-variance per loss term instead of fixed 1 / 0.5 / 0.1 / normal_weight
    log_vars = (torch.zeros(4, device=dev, requires_grad=True)
                if args.auto_loss_weight else None)
    params = list(model.parameters()) + ([log_vars] if log_vars is not None else [])
    opt = torch.optim.AdamW(params, lr=args.lr)
    start_step = 0
    ema_state = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=dev)
        if "model" in ckpt:  # new-format checkpoint: model + optim + step
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optim"])
            start_step = ckpt["step"]
            ema_state = ckpt.get("ema")
            if log_vars is not None and ckpt.get("log_vars") is not None:
                # log_vars live outside model.state_dict(); optim only holds
                # their momentum, not values — restore explicitly or they
                # silently reset to zero on resume
                with torch.no_grad():
                    log_vars.copy_(ckpt["log_vars"].to(dev))
            log(f"resumed from {args.resume} at step {start_step}")
        else:  # legacy: raw state_dict, no step/optim -> restart schedule
            model.load_state_dict(ckpt)
            log(f"resumed from {args.resume} (legacy format, step 0)")
    if ema_state is None:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def make_loader(size):
        if args.data:
            dataset, sampler = build_mixed(args.data, clip_len=args.clip_len,
                                           size=size, holdout=args.holdout)
            ld = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch, sampler=sampler,
                num_workers=args.workers, drop_last=True)
            log(f"mixed dataset: {len(dataset)} clips from "
                f"{len(args.data)} sources (size {size})")
        else:
            ld = torch.utils.data.DataLoader(
                SynthClips(size, args.clip_len), batch_size=args.batch)
        return ld

    size_schedule = parse_size_schedule(args.size_schedule, args.size)
    cur_size = size_for_step(size_schedule, start_step)
    loader = make_loader(cur_size)

    def save(path):
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                    "step": step, "ema": ema_state,
                    "log_vars": None if log_vars is None else log_vars.detach()},
                   path)

    N = (cur_size // 16) ** 2
    step = start_step
    while step < args.steps:
        new_size = size_for_step(size_schedule, step)
        if new_size != cur_size:
            cur_size = new_size
            loader = make_loader(cur_size)
            N = (cur_size // 16) ** 2
        for clip, gt, valid in loader:
            if step >= args.steps or size_for_step(size_schedule, step) != cur_size:
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
            losses = [si_log_loss(depths, gt, valid), grad_loss(depths, gt, valid),
                      temporal_loss(depths, masks), normal_loss(depths, gt, valid)]
            if log_vars is not None:
                loss = sum(torch.exp(-lv) * l + lv for lv, l in zip(log_vars, losses))
            else:
                loss = (losses[0] + 0.5 * losses[1] + 0.1 * losses[2]
                        + args.normal_weight * losses[3])
            if args.warmup > 0:  # warmup+cosine; flat lr if --warmup 0
                lr = lr_at(step, args.lr, args.steps, args.warmup)
                for g in opt.param_groups:
                    g["lr"] = lr
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if args.ema_decay > 0:
                ema_update_(ema_state, model.state_dict(), args.ema_decay)

            if step % 50 == 0:
                cur_lr = opt.param_groups[0]["lr"]
                log(f"step {step:4d}  loss {loss.item():.4f}  "
                    f"skip {skip:.2f}  lr {cur_lr:.2e}")
            step += 1
            if step % 2000 == 0:  # crash insurance for long runs
                save(work / "latest.pt")

    ckpt_path = work / "latest.pt"
    save(ckpt_path)
    log(f"saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
