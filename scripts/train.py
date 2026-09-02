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

Outputs (checkpoint latest.pt, train.log, effective config.toml) go to
work_dirs/<config name>/ (override with --work-dir).
"""
import argparse
import json
import logging
import random
import subprocess
import sys
import tomllib
from pathlib import Path

import torch

from sokkanaem import SOKKANAEM
from sokkanaem.data import SynthClips, build_mixed, even_subset
from sokkanaem.collapse import update_streak
from sokkanaem.distill import (affine_invariant_loss, dinov2_features,
                               distill_loss, load_frozen_dinov2,
                               load_frozen_teacher, teacher_disparity,
                               teacher_grad_loss)
from sokkanaem.ema import ema_update_
from sokkanaem.losses import (bin_ce_loss, boundary_location_loss,
                              dynamic_weighted_loss, edge_weighted_loss,
                              grad_loss,
                              multiscale_grad_loss, near_weighted_loss,
                              normal_loss, rank_loss, si_log_loss,
                              overshoot_loss, spread_loss, temporal_loss,
                              warp_residual_loss)
from sokkanaem.schedule import lr_at, parse_size_schedule, size_for_step


# Measured with scripts/train.py's own terms on the reported checkpoint over
# 8 mixed batches: grad_loss 1.391 -> 0.044 and normal_loss 0.0482 -> 0.0067
# when moved from metres to log depth. See --loss-space.
GRAD_LOG_GAIN, NORMAL_LOG_GAIN = 31.4, 7.2


def write_config(path, args, model_kw):
    """Write the EFFECTIVE config — the merged CLI+TOML values the run really
    used — plus the provenance to reproduce it, and return the [meta] table.

    Copying the input TOML instead recorded intent, not fact: arm1/arm2's
    config.toml still claims teacher_weight = 0.5 although both ran with the
    flag overridden to 0 (reports/20260729.md §8.3). from_checkpoint() reads
    this file, so [model] must keep its shape."""
    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        return json.dumps(str(v))

    def table(name, d):
        return ([f"\n[{name}]"] if name else []) + [
            f"{k} = {fmt(v)}" for k, v in d.items() if v is not None]

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=Path(__file__).resolve().parent)
    meta = {"command": " ".join(sys.argv),
            "git_commit": git.stdout.strip() or "unknown",
            # str(): torch.__version__ is a TorchVersion object, and this dict
            # also goes into the checkpoint — weights_only=True (the torch 2.6+
            # default) refuses to unpickle it, i.e. the run would train for
            # 14 h and then fail to load
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda or "cpu",
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "cpu")}
    path.write_text("\n".join(
        ["# effective config (merged CLI + TOML), written by train.py"]
        + table(None, {k: v for k, v in vars(args).items() if k != "config"})
        + table("model", model_kw) + table("meta", meta)) + "\n")
    return meta


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
    ap.add_argument("--workers", type=int, default=16,
                    help="DataLoader workers; the box has 32 cores and image "
                         "decode is the throughput wall, not the GPU")
    ap.add_argument("--max-skip", type=float, default=0.8,
                    help="final random-mask skip ratio")
    ap.add_argument("--no-augment", dest="augment", action="store_false",
                    help="disable clip-consistent crop/flip/colour jitter. On "
                         "by default: without it v7 hit AbsRel 0.192 / d1 0.707 "
                         "on seen clips but 0.356 / 0.519 on the holdout — the "
                         "gap was generalization, not capacity")
    ap.add_argument("--detector-mask", action="store_true",
                    help="train with detector-driven masks instead of iid "
                         "random (§4.4 mask-distribution ablation)")
    ap.add_argument("--holdout", action="append", default=None,
                    help="path substring for the val split (repeatable); "
                         "matching sequences are excluded from training")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--resume-partial", action="store_true",
                    help="load --resume non-strictly and restart the schedule "
                         "at step 0: for initializing a new architecture from "
                         "an older run's weights (v8's extra scan directions, "
                         "depthwise convs and bin head have no v7 counterpart)")
    ap.add_argument("--work-dir", default=None,
                    help="output dir; default work_dirs/<config name>")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds torch and random (DataLoader workers derive "
                         "theirs from torch's). The 3-arm probe ran unseeded, "
                         "so sampler order, augmentation and random masks all "
                         "differed between arms — small gaps like bin CE's "
                         "were not separable from seed variance")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="eval-time shadow-weight EMA decay (0 disables)")
    ap.add_argument("--auto-loss-weight", action="store_true",
                    help="Kendall multi-task uncertainty weighting instead "
                         "of the fixed si_log/grad/temporal/normal weights")
    ap.add_argument("--msgrad-weight", type=float, default=0.0,
                    help="MiDaS-style multi-scale gradient matching loss on "
                         "normalized disparity (0 = off). Single-scale grad "
                         "only sees 1-pixel edges; the pyramid also penalizes "
                         "low-frequency shape error")
    ap.add_argument("--normal-weight", type=float, default=0.0,
                    help="surface-normal loss weight (ignored if "
                         "--auto-loss-weight; 0 = off, matches old default)")
    ap.add_argument("--warp-weight", type=float, default=0.0,
                    help="flow-warped residual loss, the training-time version "
                         "of the TCE metric (0 = off). Costs one RAFT-small "
                         "forward per clip per step")
    ap.add_argument("--spread-weight", type=float, default=0.0,
                    help="penalise dynamic-range compression: match the "
                         "std of predicted log depth to the GT's, per sample "
                         "(REPORT 4.32 measured 0.47x range on Bonn). Scale-"
                         "free and symmetric, so it cannot be bought by "
                         "inflating the range with noise")
    ap.add_argument("--loss-space", choices=("metric", "log"), default="metric",
                    help="space the gradient and normal terms are computed in. "
                         "'metric' is what the reported checkpoint used: both "
                         "take raw metres, so on 1.5-4 m indoor footage "
                         "dz/dx ~ 0.01 and the pseudo-normal is (0,0,1) "
                         "everywhere -- the term that exists for boundaries "
                         "contributes almost nothing, and the far-range "
                         "synthetic sources own the gradient term as well. "
                         "'log' puts both in log depth, which is scale-free, "
                         "and also normalizes the multiscale term per frame "
                         "rather than per batch (MiDaS's own rule).")
    ap.add_argument("--edge-weight", type=float, default=0.0,
                    help="GT-depth-gradient weighted log L1, aimed at "
                         "foreground objects at depth discontinuities (0 = off)")
    ap.add_argument("--near-weight", type=float, default=0.0,
                    help="close-range weighted log L1 (PLAN_ACC A7). REPORT "
                         "§4.43 puts our error inside 2 m at 2.38x a 343M "
                         "reference while the rest of the frame sits at 1.13x, "
                         "and 73%% of the dynamic pixels are in that band "
                         "(0 = off)")
    ap.add_argument("--near-band", type=float, default=2.0,
                    help="metres; the depth below which --near-weight applies")
    ap.add_argument("--near-gain", type=float, default=4.0,
                    help="relative weight inside the band (1 = no reweighting)")
    ap.add_argument("--size-schedule", default=None,
                    help="progressive resolution curriculum: "
                         "'step:size,step:size,...' e.g. '0:128,20000:256' "
                         "(default: fixed --size throughout)")
    ap.add_argument("--warmup", type=int, default=2000,
                    help="LR linear-warmup steps before cosine decay "
                         "(--warmup 0 disables the whole schedule -> flat lr)")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="max grad norm (0 disables)")
    ap.add_argument("--collapse-patience", type=int, default=1000,
                    help="abort if predicted depth std stays under "
                         "--collapse-eps for this many consecutive steps "
                         "(0 disables; catches constant-output collapse "
                         "early instead of burning the full run on it)")
    ap.add_argument("--collapse-eps", type=float, default=1e-4)
    ap.add_argument("--distill-weight", type=float, default=0.0,
                    help="feature-distillation loss weight from a frozen "
                         "DINOv2 encoder (0 = off; needs --data, not "
                         "--detector-mask). Zero inference cost — the "
                         "frozen model only runs during training.")
    ap.add_argument("--distill-model", default="facebook/dinov2-small")
    ap.add_argument("--teacher-weight", type=float, default=0.0,
                    help="output-level distillation from a frozen relative-"
                         "depth teacher (0 = off). Zero inference cost; "
                         "affine-invariant, so it supervises geometry without "
                         "touching the metric scale the GT provides")
    ap.add_argument("--teacher-grad-weight", type=float, default=0.0,
                    help="distil only the teacher's disparity GRADIENTS, over a "
                         "resolution pyramid. --teacher-weight matches its "
                         "values and was measured to hurt both domains "
                         "(configs/main_v8.toml); gradients carry the "
                         "sharpness without pinning the depth range, which is "
                         "the failure that removed the value term.")
    ap.add_argument("--teacher-model",
                    default="depth-anything/Depth-Anything-V2-Small-hf")
    # capacity knobs on the CLI: [model] used to be TOML-only, which meant a
    # capacity probe needed a new config file per arm. write_config records
    # whatever lands in model_kw, so from_checkpoint still rebuilds the arm.
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--d-state", type=int, default=None)
    ap.add_argument("--dec-width", type=int, default=None,
                    help="DPTDecoder fusion width (default 64)")
    ap.add_argument("--fuse-norm", action="store_true", default=None,
                    help="DPT-style normalized fusion: one GroupNorm per branch "
                         "before the decoder's skip add, so the backbone arm "
                         "stops drowning the RGB arm 5-10x. Rejected as a solo "
                         "arm (REPORT 4.44's A); worth re-testing under a loss "
                         "that asks for boundary detail")
    ap.add_argument("--full-res", action="store_true", default=None,
                    help="D1: learned full-resolution upsampling (zero-init "
                         "pixel-shuffle residual + full-res RGB detail) "
                         "instead of the final bilinear 2x")
    ap.add_argument("--train-clips", type=int, default=None,
                    help="train on this many clips, spread evenly (saturation "
                         "probes measure capacity on a small hard subset, not "
                         "generalization on 200k clips)")
    ap.add_argument("--val-split", action="store_true",
                    help="train on the --holdout MATCHES instead of the rest: "
                         "a saturation probe deliberately fits the clips it is "
                         "scored on. Never use with the sealed real holdout.")
    ap.add_argument("--boundary-weight", type=float, default=0.0,
                    help="PLAN Stage A L_boundary_location: L1 on the "
                         "prediction's gradient outside the GT boundary band, "
                         "plus a one-sided hinge to reach the GT's gradient "
                         "inside it. Targets boundary precision / flat TV / "
                         "overshoot, which the symmetric grad terms do not see")
    ap.add_argument("--q-teacher", default=None,
                    help="PLAN §7 Q2: distil from a Q checkpoint (frozen DA2 "
                         "shape + metric calibration). Unlike the rejected "
                         "teacher arms, this teacher is METRIC and stable "
                         "(zero alignment failures under the median gauge), so "
                         "it can be matched in depth space instead of through "
                         "an affine-invariant fit -- and it is DENSE, which is "
                         "the sharp indoor supervision our Kinect GT cannot "
                         "give (REPORT 4.44/4.45)")
    ap.add_argument("--q-weight", type=float, default=0.5,
                    help="si-log against the teacher's depth")
    ap.add_argument("--q-grad-weight", type=float, default=0.5,
                    help="multiscale gradient match against the teacher -- the "
                         "term that carries its boundaries")
    ap.add_argument("--shape-on-synthetic", action="store_true",
                    help="Depth Anything V2's central choice, ported: run the "
                         "SHAPE terms (grad, multiscale grad, boundary, rank) "
                         "only on sources whose GT is synthetic, and keep the "
                         "metric terms on everything. Kinect GT smears every "
                         "silhouette, so supervising shape with it teaches the "
                         "blur we are trying to remove")
    ap.add_argument("--trim-real-band", action="store_true",
                    help="drop the GT boundary band from the metric terms on "
                         "REAL sources (DA V2 discards its noisiest pseudo-"
                         "label pixels for the same reason). Needs "
                         "--shape-on-synthetic's source tags")
    ap.add_argument("--overshoot-weight", type=float, default=0.0,
                    help="penalty for leaving the local GT range at a boundary "
                         "-- the ringing the one-sided boundary hinge permits")
    ap.add_argument("--dynamic-weight", type=float, default=0.0,
                    help="log-depth L1 weighted towards independently-moving "
                         "pixels (flow residual vs the frame's dominant "
                         "motion). Targets the 2.66x dynamic-pixel gap; costs "
                         "one RAFT pass per batch")
    ap.add_argument("--dynamic-gain", type=float, default=4.0)
    ap.add_argument("--rank-weight", type=float, default=0.0,
                    help="PLAN Stage A L_pairwise_order: logistic ordering "
                         "loss on random pixel pairs, gauge-free")
    ap.add_argument("--bin-weight", type=float, default=0.0,
                    help="soft cross-entropy on the depth-bin distribution "
                         "(0 = off; needs a bins>0 dpt decoder). Without it "
                         "the head is supervised only through its expectation "
                         "and degenerates into scalar regression behind a "
                         "softmax — measured entropy 0.77 of maximum, "
                         "identical at boundaries and in flat regions")

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
    for flag, key in (("dim", "dim"), ("depth", "depth"),
                      ("d_state", "d_state"), ("dec_width", "dec_width"),
                      ("full_res", "full_res"), ("fuse_norm", "fuse_norm")):
        v = getattr(args, flag)
        if v is not None:
            model_kw[key] = v

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    name = Path(args.config).stem if args.config else "default"
    work = Path(args.work_dir or f"work_dirs/{name}")
    work.mkdir(parents=True, exist_ok=True)
    meta = write_config(work / "config.toml", args, model_kw)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(work / "train.log")])
    log = logging.getLogger("train").info
    log(f"work dir: {work}  seed {args.seed}  commit {meta['git_commit'][:8]}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SOKKANAEM(**model_kw).to(dev)
    # Feature distillation (zero inference cost — frozen, training-only):
    # import DINOv2's pretrained visual features into our from-scratch
    # encoder via a trainable projection head, matched by cosine loss.
    dinov2 = load_frozen_dinov2(args.distill_model, dev) if args.distill_weight > 0 else None
    teacher = (load_frozen_teacher(args.teacher_model, dev)
               if args.teacher_weight > 0 or args.teacher_grad_weight > 0 else None)
    # bin logits are internal to the decoder head; a forward hook collects one
    # entry per frame (forward_clip calls the decoder T times) without
    # threading them through every return signature
    q_teacher = None
    if args.q_teacher:
        from sokkanaem import from_checkpoint as _load
        q_teacher = _load(args.q_teacher, dev).eval()
        for prm in q_teacher.parameters():
            prm.requires_grad_(False)
        log(f"Q teacher: {args.q_teacher} "
            f"({sum(p.numel() for p in q_teacher.parameters())/1e6:.1f}M, frozen)")
    bin_logits = []
    if args.bin_weight > 0:
        assert getattr(model.decoder, "bins", 0), \
            "--bin-weight needs decoder = 'dpt' with bins > 0"
        model.decoder.head.register_forward_hook(
            lambda mod, inp, out: bin_logits.append(out))
    distill_proj = (torch.nn.Linear(model.dim, dinov2.config.hidden_size).to(dev)
                    if dinov2 is not None else None)
    # Kendall multi-task uncertainty weighting (§ auto-loss-weight): learnable
    # log-variance per loss term instead of fixed 1 / 0.5 / 0.1 / normal_weight
    log_vars = (torch.zeros(4, device=dev, requires_grad=True)
                if args.auto_loss_weight else None)
    params = (list(model.parameters())
              + ([log_vars] if log_vars is not None else [])
              + (list(distill_proj.parameters()) if distill_proj is not None else []))
    opt = torch.optim.AdamW(params, lr=args.lr)
    start_step = 0
    ema_state = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=dev)
        if args.resume_partial:
            # new-arch init: take whatever weights match, leave the rest at
            # their fresh init, and start the schedule from scratch
            sd = ckpt.get("ema") or ckpt.get("model") or ckpt
            # strict=False only forgives missing/unexpected KEYS, not changed
            # shapes (bins 64 -> 128 is three of those), so drop those here.
            # log_range is special-cased: it is a buffer holding the config's
            # [d_min, d_max] and its shape always matches, so restoring it
            # would silently cancel a d_max change.
            cur = model.state_dict()
            sd = {k: v for k, v in sd.items()
                  if k != "decoder.log_range"
                  and k in cur and cur[k].shape == v.shape}
            miss, extra = model.load_state_dict(sd, strict=False)
            log(f"partial init from {args.resume}: {len(miss)} new tensors "
                f"kept at init, {len(extra)} checkpoint tensors unused")
        elif "model" in ckpt:  # new-format checkpoint: model + optim + step
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
            if distill_proj is not None and ckpt.get("distill_proj") is not None:
                distill_proj.load_state_dict(ckpt["distill_proj"])
            log(f"resumed from {args.resume} at step {start_step}")
        else:  # legacy: raw state_dict, no step/optim -> restart schedule
            model.load_state_dict(ckpt)
            log(f"resumed from {args.resume} (legacy format, step 0)")
    if ema_state is None:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def make_loader(size):
        if args.data:
            dataset, sampler = build_mixed(args.data, clip_len=args.clip_len,
                                           size=size, holdout=args.holdout,
                                           val=args.val_split,
                                           tag_source=tag_source,
                                           augment=args.augment)
            if args.train_clips:
                # the per-source equalizing sampler indexes the full concat, so
                # it cannot survive the subset -- shuffle instead
                dataset, sampler = even_subset(dataset, args.train_clips), None
            ld = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch, sampler=sampler,
                shuffle=sampler is None,
                num_workers=args.workers, drop_last=True,
                # measured: throughput was ~48 decoded frames/s regardless of
                # resolution, i.e. PNG/JPEG decode from /archive was the wall,
                # not the GPU (0% util at 128px). Keep workers warm across the
                # resolution-curriculum loader rebuilds and prefetch deeper.
                persistent_workers=args.workers > 0,
                prefetch_factor=4 if args.workers > 0 else None,
                pin_memory=True)
            log(f"mixed dataset: {len(dataset)} clips from "
                f"{len(args.data)} sources (size {size})")
        else:
            ld = torch.utils.data.DataLoader(
                SynthClips(size, args.clip_len), batch_size=args.batch)
        return ld

    tag_source = args.shape_on_synthetic or args.trim_real_band
    size_schedule = parse_size_schedule(args.size_schedule, args.size)
    cur_size = size_for_step(size_schedule, start_step)
    loader = make_loader(cur_size)

    def save(path):
        # provenance travels with the weights: a checkpoint that outlives its
        # work dir still names its commit, seed and exact arguments
        torch.save({"meta": {**meta, "args": vars(args), "model": model_kw},
                    "model": model.state_dict(), "optim": opt.state_dict(),
                    "step": step, "ema": ema_state,
                    "log_vars": None if log_vars is None else log_vars.detach(),
                    "distill_proj": None if distill_proj is None else distill_proj.state_dict()},
                   path)

    N = (cur_size // 16) ** 2
    step = start_step
    low_std_streak = 0
    while step < args.steps:
        new_size = size_for_step(size_schedule, step)
        if new_size != cur_size:
            cur_size = new_size
            loader = make_loader(cur_size)
            N = (cur_size // 16) ** 2
        for batch in loader:
            if step >= args.steps or size_for_step(size_schedule, step) != cur_size:
                break
            clip, gt, valid = (x.to(dev) for x in batch[:3])
            B, T = clip.shape[:2]
            # per-sample: 1 where the GT is synthetic (sharp at boundaries)
            synth = (batch[3].to(dev).view(B, 1, 1, 1, 1)
                     if tag_source else None)
            # what the shape terms are allowed to see, and what the metric
            # terms are allowed to see. Defaults keep every pixel of both.
            v_shape = valid if synth is None or not args.shape_on_synthetic \
                else valid * synth
            v_metric = valid
            if args.trim_real_band and synth is not None:
                from sokkanaem.sharpness import boundary_band
                shp = (-1, 1) + tuple(gt.shape[-2:])
                band = boundary_band(gt.reshape(shp),
                                     valid.reshape(shp)).reshape(gt.shape)
                # real sources only: their boundary band is where the sensor
                # is least trustworthy, and it is 10% of the pixels
                v_metric = valid * (1 - (1 - synth) * band.float())
            bin_logits.clear()

            tokens = None
            if args.detector_mask:
                skip = 0.0
                depths, masks = model.forward_clip(clip)  # detector-driven
            else:
                # random-mask scheduling: skip ratio ramps 0 -> max (§3.2)
                skip = args.max_skip * min(1.0, step / max(1, args.steps // 2))
                fm = (torch.rand(B, T, N, device=dev) > skip).float()
                fm[:, 0] = 1.0  # first frame always full
                if distill_proj is not None:
                    depths, masks, tokens = model.forward_clip(
                        clip, force_mask=fm, return_tokens=True)
                else:
                    depths, masks = model.forward_clip(clip, force_mask=fm)
            # the gradient and normal terms are scale-dependent, so the space
            # they see decides whether indoor footage can reach them at all
            per_sample = args.loss_space == "log"
            gp, gg = ((depths.clamp(min=1e-3).log(), gt.clamp(min=1e-3).log())
                      if per_sample else (depths, gt))
            losses = [si_log_loss(depths, gt, v_metric),
                      grad_loss(gp, gg, v_shape),
                      temporal_loss(depths, masks),
                      normal_loss(gp, gg, v_shape)]
            if log_vars is not None:
                loss = sum(torch.exp(-lv) * l + lv for lv, l in zip(log_vars, losses))
            else:
                # A log-space gradient is 31x smaller than a metric one and a
                # log-space normal 7.2x smaller (medians over 8 real mixed
                # batches under the reported checkpoint). Switching space
                # without compensating would down-weight both terms by those
                # factors, so the arm would measure the down-weighting rather
                # than the space. These gains hold each term's starting
                # contribution fixed; what changes is which sources reach it.
                gw = 0.5 * (GRAD_LOG_GAIN if per_sample else 1.0)
                nw = args.normal_weight * (NORMAL_LOG_GAIN if per_sample else 1.0)
                loss = (losses[0] + gw * losses[1] + 0.1 * losses[2]
                        + nw * losses[3])
            if args.msgrad_weight > 0:
                loss = loss + args.msgrad_weight * multiscale_grad_loss(
                    depths, gt, v_shape, per_sample=per_sample)
            if args.warp_weight > 0:
                loss = loss + args.warp_weight * warp_residual_loss(
                    clip, depths, gt, valid)
            if args.edge_weight > 0:
                loss = loss + args.edge_weight * edge_weighted_loss(
                    depths, gt, v_shape)
            if args.near_weight > 0:
                loss = loss + args.near_weight * near_weighted_loss(
                    depths, gt, v_metric, args.near_band, args.near_gain)
            if args.boundary_weight > 0:
                loss = loss + args.boundary_weight * boundary_location_loss(
                    depths, gt, v_shape)
            if args.dynamic_weight > 0:
                loss = loss + args.dynamic_weight * dynamic_weighted_loss(
                    clip, depths, gt, v_metric, args.dynamic_gain)
            if args.overshoot_weight > 0:
                loss = loss + args.overshoot_weight * overshoot_loss(
                    depths, gt, v_shape)
            if args.rank_weight > 0:
                loss = loss + args.rank_weight * rank_loss(depths, gt, v_shape)
            if q_teacher is not None:
                with torch.no_grad():
                    tgt, _ = q_teacher.forward_clip(clip)
                # every pixel, including where the Kinect GT has a hole: the
                # teacher's value there is the only supervision available
                ones = torch.ones_like(tgt)
                if args.q_weight > 0:
                    loss = loss + args.q_weight * si_log_loss(depths, tgt, ones)
                if args.q_grad_weight > 0:
                    loss = loss + args.q_grad_weight * multiscale_grad_loss(
                        depths, tgt, ones, per_sample=True)
            if args.spread_weight > 0:
                loss = loss + args.spread_weight * spread_loss(
                    depths, gt, valid)
            if args.bin_weight > 0:
                # hook order is frame-major (t0 batch, t1 batch, ...), so the
                # targets must be transposed to match before flattening
                lg = torch.cat(bin_logits, 0)
                g = gt.transpose(0, 1).reshape(B * T, 1, *gt.shape[-2:])
                v = valid.transpose(0, 1).reshape(B * T, 1, *valid.shape[-2:])
                loss = loss + args.bin_weight * bin_ce_loss(
                    lg, model.decoder.bin_centres(), g, v)
            if teacher is not None:
                # dense target on every pixel, including where the Kinect GT
                # has holes — both terms are gauge-free so neither fights the
                # GT supervision over scale
                flat = clip.reshape(B * T, 3, *clip.shape[-2:])
                tdisp = teacher_disparity(teacher, flat)
                dflat = depths.reshape(B * T, 1, *depths.shape[-2:])
                if args.teacher_weight > 0:
                    loss = loss + args.teacher_weight * affine_invariant_loss(
                        dflat, tdisp)
                if args.teacher_grad_weight > 0:
                    loss = loss + args.teacher_grad_weight * teacher_grad_loss(
                        dflat, tdisp)
            if distill_proj is not None and tokens is not None:
                gh, gw = clip.shape[-2] // 16, clip.shape[-1] // 16
                frames_flat = clip.reshape(B * T, 3, *clip.shape[-2:])
                target = dinov2_features(dinov2, frames_flat, (gh, gw))
                dloss = distill_loss(tokens.reshape(B * T, N, -1), distill_proj, target)
                loss = loss + args.distill_weight * dloss
            if args.warmup > 0:  # warmup+cosine; flat lr if --warmup 0
                lr = lr_at(step, args.lr, args.steps, args.warmup)
                for g in opt.param_groups:
                    g["lr"] = lr
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if log_vars is not None:
                # unconstrained log-variance drifts monotonically negative
                # when a task's loss keeps shrinking (observed: temporal
                # term -2.3 -> -5.2 over 10k steps) -> exp(-lv) blows up.
                # [-8, 8] keeps the adaptive weighting within a sane range
                # (4e-4x to 3000x) without ever reaching fp32 overflow.
                with torch.no_grad():
                    log_vars.clamp_(-8, 8)
            if args.ema_decay > 0:
                ema_update_(ema_state, model.state_dict(), args.ema_decay)

            depth_std = depths.std().item()
            if step > 200:
                low_std_streak = update_streak(low_std_streak, depth_std, args.collapse_eps)
            if step % 50 == 0:
                cur_lr = opt.param_groups[0]["lr"]
                log(f"step {step:4d}  loss {loss.item():.4f}  "
                    f"skip {skip:.2f}  lr {cur_lr:.2e}  depth_std {depth_std:.4f}")
            if args.collapse_patience > 0 and low_std_streak >= args.collapse_patience:
                log(f"ABORT: predicted depth std < {args.collapse_eps} for "
                    f"{low_std_streak} consecutive steps -> constant-output "
                    f"collapse (root cause is almost always a mis-weighted "
                    f"loss term with a trivial degenerate minimizer)")
                save(work / "latest.pt")
                raise SystemExit(1)
            step += 1
            if step % 2000 == 0:  # crash insurance for long runs
                save(work / "latest.pt")

    ckpt_path = work / "latest.pt"
    save(ckpt_path)
    log(f"saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
