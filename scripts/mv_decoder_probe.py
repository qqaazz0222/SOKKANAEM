"""Decoder-only probe: does an ImageNet-pretrained MambaVision backbone close the
native model's depth-shape gap? 2026-09-15.

Runs in the ISOLATED `mambavision` env (see
paper/SOKKANAEM_MAMBAVISION_STRUCTURE_CHECK_20260915.md); the main env's
transformers pin would break there. Scoring runs in the main env:
scripts/mv_decoder_probe_score.py.

Two arms share everything except the frozen backbone:
  mambavision  MambaVision-T stem..stage 2 at 256 px with global windows
               [8, 8, 16, 8] and ImageNet-1K weights. Stage-2 blocks 2, 4, 6, 8
               (320x16x16) feed the decoder's four inputs.
  native       the 4.19M native backbone (d1-boundary3 EMA), every patch active,
               no temporal state carried between frames (the repository measured
               per-frame reset equal to dense in accuracy). Its four block outputs
               (192x16x16) feed the decoder. This arm is the control for the recipe.

Decoder: the native DPTDecoder (bins 64, full-res RGB detail), with every weight
transplanted from the native checkpoint EXCEPT the four input projections, which
are freshly initialised in BOTH arms so neither starts with a matched projection.

Training: decoder only, the native loss set and weights of the d1-boundary3 run
(si-log, grad 0.5, temporal 0.1, normal 0.05, msgrad 0.5, warp 2.0, edge 2.0,
boundary 3.0, spread 0.5, bin CE 0.2, metric loss space), the same data mix and
holdout (configs/main_v8.toml), batch 4, 4-frame clips, 256 px, AdamW lr 1e-4,
500 warmup + cosine, grad clip 1.0, EMA 0.999, seed 0. BEHAVE is never read during
training.

Caveat by design: the native backbone was trained end to end with its decoder for
tens of thousands of steps; here both arms get the same short decoder-only budget,
so the native arm is a recipe control, not the native model's best result.

Calibration follow-up (`calibrate` mode): Q0's per-frame metric head
(sokkanaem/qmodel.py) is added on top of each trained probe, reading the
backbone's pooled last-tap tokens, with backbone and decoder frozen. Q0's recipe
is reproduced: real sources only (tum, bonn), batch 2, 4-frame clips, 256 px,
AdamW lr 1e-3, si-log loss, grad clip 1.0, no EMA, seed 0. train_q.py passes
(warmup, steps) into lr_at's (total, warmup) slots, so Q0 actually trained with
a linear ramp from 0 to 1e-3 across all 4000 steps; that effective schedule is
what is reproduced here.

Usage:
  python scripts/mv_decoder_probe.py train     --backbone {mambavision,native} [--steps N]
  python scripts/mv_decoder_probe.py predict   --backbone {mambavision,native} [--frames N] [--calibrated]
  python scripts/mv_decoder_probe.py calibrate --backbone {mambavision,native} [--steps N]
"""
import argparse
import json
import random
import statistics
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sokkanaem import from_checkpoint
from sokkanaem.data import build_mixed
from sokkanaem.ema import ema_update_
from sokkanaem.losses import (bin_ce_loss, boundary_location_loss, edge_weighted_loss,
                              grad_loss, multiscale_grad_loss, normal_loss, si_log_loss,
                              spread_loss, temporal_loss, warp_residual_loss)
from sokkanaem.model import DPTDecoder
from sokkanaem.schedule import lr_at
from sokkanaem.sharpness import norm_field
import torch.nn.functional as F

NATIVE = ROOT / "work_dirs/d1-boundary3-s0/latest.pt"
MV_WEIGHTS = ROOT / "work_dirs/mambavision_20260915/mambavision_tiny_1k.pth.tar"
OUT = ROOT / "work_dirs/mv_decoder_probe_20260915"
from sokkanaem.losses import overshoot_loss
from sokkanaem.qflat_state import flat_band_loss, flat_excess_loss, flat_mask, flat_plane_loss

DISTILLED = ROOT / "work_dirs/mv_da2_distill_20260915/student.pt"
DISTILLED_HR = ROOT / "work_dirs/mv_da2_distill_hr_20260915/student.pt"
DISTILLED_MR = ROOT / "work_dirs/mv_da2_distill_mr_20260916/student.pt"
TAPS = (1, 3, 5, 7)
BEHAVE = {"Date01_Sub01_basketball.1": ("work_dirs/qbehave_development_ready2_20260911", 622, 878),
          "Date01_Sub01_keyboard_move.1": ("work_dirs/qbehave_development_ready2_20260911", 622, 878),
          "Date02_Sub02_backpack_back.1": ("work_dirs/qbehave_pilot_eval_20260910", 622, 878),
          "Date02_Sub02_backpack_hand.1": ("work_dirs/qbehave_pilot_eval_20260910", 622, 878),
          "Date03_Sub03_backpack_back.1": ("work_dirs/qbehave_holdout_ready_20260911", 0, 1024),
          "Date04_Sub05_backpack.1": ("work_dirs/qbehave_holdout_ready_20260911", 0, 1024)}


class MambaVisionBackbone(torch.nn.Module):
    dim = 320
    student = None   # a distilled backbone state_dict replacing the ImageNet weights
    size = 256       # input size the global stage-2 window is built for; frames are resized to it

    def __init__(self):
        super().__init__()
        import mambavision.models.mamba_vision as mv
        self.window_partition = mv.window_partition
        # drop_path_rate=0: stochastic depth is inert in eval and has no parameters,
        # so probe outputs are unchanged, and a distillation student trains without it
        model = mv.mamba_vision_T(pretrained=False, window_size=[8, 8, self.size // 16, 8], drop_path_rate=0.0)
        if self.student is None:
            state = torch.load(MV_WEIGHTS, map_location="cpu", weights_only=False)["state_dict"]
        else:
            state = torch.load(self.student, map_location="cpu", weights_only=False)["backbone"]
        model.load_state_dict(state, strict=True)
        self.model = model.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, frames):
        return self.features(frames)

    def features(self, frames):
        """The four stage-2 taps; differentiable, for a distillation student."""
        m = self.model
        if frames.shape[-1] != self.size:
            frames = F.interpolate(frames, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = m.levels[1](m.levels[0](m.patch_embed((frames - self.mean) / self.std)))
        level = m.levels[2]
        n, c, h, w = x.shape
        if not h == w == level.window_size:
            raise ValueError("expects one global stage-2 window, i.e. 256 px input")
        x = self.window_partition(x, level.window_size)      # (n, h*w, c), raster order
        feats = []
        for i, block in enumerate(level.blocks):
            x = block(x)
            if i in TAPS:
                feats.append(x.transpose(1, 2).reshape(n, c, h, w))
        return feats


class NativeBackbone(torch.nn.Module):
    dim = 192
    size = 256

    def __init__(self):
        super().__init__()
        self.model = from_checkpoint(NATIVE, "cpu", tau_on=-1., tau_off=-1.).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, frames):
        m = self.model
        n, grid = frames.shape[0], frames.shape[-1] // m.p
        tokens = m.embed(frames).flatten(2).transpose(1, 2)
        mask = torch.ones(n, grid * grid, device=frames.device)
        # hs=None: a fresh temporal state, so every frame is processed independently
        _, _, _, _, feats = m._forward_tokens(tokens, mask, None, (grid, grid))
        return [t.transpose(1, 2).reshape(n, m.dim, grid, grid) for t in feats]


class DistilledMambaVisionBackbone(MambaVisionBackbone):
    """The same architecture, weights from scripts/mv_da2_distill.py."""
    student = DISTILLED


class HighResDistilledBackbone(MambaVisionBackbone):
    """Distilled at 592 px, so its 37x37 stage-2 grid matches DA2's without any resize.

    Frames are upsampled from 256 px as Q0 upsamples them to 518 px for DA2; the depth
    comes back to the caller's resolution. The student path is set from --distilled-hr.
    """
    size = 592
    student = DISTILLED_HR


class MidResDistilledBackbone(MambaVisionBackbone):
    """Distilled at 384 px: a 24x24 stage-2 grid, between the 16x16 and 37x37 arms.

    Appendix I.20 located the 592 px arms' excess flat TV in the 4-16 px band, which is the
    37x37 grid's own token footprint (~6.9 px at the 256 px output); the 16x16 grid cannot
    represent that band at all and is quiet because it is blunt. 384 px puts one token at
    ~10.7 px. The student path is set from --distilled-mr.
    """
    size = 384
    student = DISTILLED_MR


class FineTunedBackbone(MambaVisionBackbone):
    """The 256 px distilled student after joint depth fine-tuning; path set from --out."""
    student = None


class FineTunedHighResBackbone(MambaVisionBackbone):
    """The 592 px distilled student after joint depth fine-tuning; path set from --out."""
    size = 592
    student = None


class FineTunedMidResBackbone(MambaVisionBackbone):
    """The 384 px distilled student after joint depth fine-tuning; path set from --out."""
    size = 384
    student = None


BACKBONES = {"mambavision": MambaVisionBackbone, "native": NativeBackbone,
             "mambavision_da2": DistilledMambaVisionBackbone,
             "mambavision_da2_hr": HighResDistilledBackbone,
             "mambavision_da2_mr": MidResDistilledBackbone,
             "mambavision_da2_mr_ft": FineTunedMidResBackbone,
             "mambavision_da2_ft": FineTunedBackbone,
             "mambavision_da2_hr_ft": FineTunedHighResBackbone}


class CalibrationHead(torch.nn.Module):
    """Q0's per-frame metric head, identical layers, init and output mapping.

    Reads the backbone's pooled tokens and maps the decoder's per-frame normalized
    disparity to metres: disp = softplus(scale) * dn + shift, depth = 1 / disp.
    """

    def __init__(self, dim, d_min=0.3, d_max=150.0):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, 64),
                                       torch.nn.GELU(), torch.nn.Linear(64, 2))
        torch.nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([-1.5, 0.3]))
        self.d_min, self.d_max = d_min, d_max

    def forward(self, depth, pooled):
        disp = 1.0 / depth.clamp(min=1e-3)
        normalized = norm_field(disp, torch.ones_like(disp))
        scale, shift = self.net(pooled.float()).unbind(-1)
        metric = F.softplus(scale).view(-1, 1, 1, 1) * normalized + shift.view(-1, 1, 1, 1)
        return (1.0 / metric.clamp(min=1.0 / self.d_max)).clamp(min=self.d_min, max=self.d_max)


def frozen_probe(backbone_name, out):
    """The trained probe (EMA decoder) with every parameter frozen."""
    checkpoint = torch.load(Path(out) / backbone_name / "decoder.pt", map_location="cpu", weights_only=False)
    backbone = BACKBONES[backbone_name]().cuda()
    decoder = build_decoder(backbone.dim)
    decoder.load_state_dict(checkpoint["ema"], strict=True)
    return backbone, decoder.cuda().eval().requires_grad_(False), checkpoint["step"]


def calibrate(args):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    arm = Path(args.out) / args.backbone
    if (arm / "calib.pt").exists():
        raise SystemExit(f"{arm / 'calib.pt'} exists; use a new directory")
    backbone, decoder, decoder_step = frozen_probe(args.backbone, args.out)
    head = CalibrationHead(backbone.dim).cuda().train()
    config = tomllib.loads((ROOT / "configs/main_v8.toml").read_text())
    dataset, sampler = build_mixed(config["data"][:2], clip_len=4, size=256,
                                   holdout=config["holdout"], augment=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler,
                                         num_workers=args.workers, drop_last=True,
                                         persistent_workers=args.workers > 0, pin_memory=True)
    params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    log = open(arm / "calib.log", "a")
    step, start = 0, time.time()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            clip, gt, valid = (x.cuda(non_blocking=True) for x in batch[:3])
            B, T = clip.shape[:2]
            frames = clip.reshape(B * T, *clip.shape[2:])
            with torch.no_grad():
                feats = backbone(frames)
                depth = depth_at(decoder, feats, frames, backbone.size)
            out = head(depth, feats[-1].mean((2, 3))).view(B, T, *depth.shape[1:])
            loss = si_log_loss(out, gt, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            for group in opt.param_groups:
                group["lr"] = 1e-3 * (step + 1) / args.steps      # Q0's effective schedule
            opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                line = (f"step {step} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} "
                        f"median {out.median().item():.2f} m elapsed {time.time() - start:.0f}s")
                print(line, flush=True)
                log.write(line + "\n"); log.flush()
    torch.save({"head": head.state_dict(), "step": step, "decoder_step": decoder_step,
                "meta": {"backbone": args.backbone, "recipe": "Q0 calibration, train_q.py --real-only",
                         "args": vars(args)}}, arm / "calib.pt")
    print("CALIB_DONE", args.backbone, step, flush=True)


def build_decoder(dim, seed=0):
    reference = from_checkpoint(NATIVE, "cpu").decoder
    torch.manual_seed(seed)
    decoder = DPTDecoder(dim, patch_size=16, width=reference.width, bins=reference.bins,
                         fuse_norm=reference.norm_x is not None, full_res=reference.full_res)
    decoder.set_n_blocks(4)
    state = {k: v for k, v in reference.state_dict().items() if not k.startswith("proj.")}
    missing, unexpected = decoder.load_state_dict(state, strict=False)
    if unexpected or not all(k.startswith("proj.") for k in missing):
        raise RuntimeError(f"decoder transplant mismatch: {missing} {unexpected}")
    return decoder


def depth_at(decoder, feats, frames, size):
    """Decoder depth with the frames resized to the backbone's input size, returned at
    the frames' size. A no-op at 256 px, so every 256 px arm computes what it did before."""
    if frames.shape[-1] == size:
        return decoder(feats, frames)
    wide = F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)
    return F.interpolate(decoder(feats, wide), size=frames.shape[-2:], mode="bilinear", align_corners=False)


def native_depth_loss(decoder, feats, frames, clip, gt, valid, bin_logits, size,
                      boundary=3.0, overshoot=0.0, flat=0.0, synthetic=None, flat_mode="excess",
                      flat_win=9, flat_win_hi=17):
    """The d1-boundary3 loss set on frame-major decoder input. Returns (depths, loss).

    `overshoot` adds sokkanaem.losses.overshoot_loss (the gate metric's differentiable form);
    `flat` adds the excess normalized-disparity gradient over the GT's own in GT-flat regions
    (sokkanaem.qflat_state.flat_excess_loss with the GT as reference). `synthetic`, a (B*T,1,1,1)
    frame-major bool, restricts that term to synthetic-GT frames: real sensor labels are quantized
    flat (92.5 % exact-zero gradient inside the flat mask, 18.6 % on synthetic), which pulls
    slanted planes flat. `flat_mode="plane"` swaps that term for the local plane-fit residual
    (flat_plane_loss), which leaves any slope free and pays only for curvature the GT lacks.
    `flat_mode="band"` swaps it for a difference-of-box bandpass (flat_band_loss) restricted to
    the flat_win..flat_win_hi pixel band, leaving both finer detail and coarser slope untouched."""
    B, T = clip.shape[:2]
    out = depth_at(decoder, feats, frames, size)
    depths = out.view(T, B, *out.shape[1:]).transpose(0, 1)
    masks = torch.ones(B, T, (clip.shape[-2] // 16) * (clip.shape[-1] // 16), device=clip.device)
    loss = (si_log_loss(depths, gt, valid) + 0.5 * grad_loss(depths, gt, valid)
            + 0.1 * temporal_loss(depths, masks) + 0.05 * normal_loss(depths, gt, valid)
            + 0.5 * multiscale_grad_loss(depths, gt, valid, per_sample=False)
            + 2.0 * warp_residual_loss(clip, depths, gt, valid)
            + 2.0 * edge_weighted_loss(depths, gt, valid)
            + boundary * boundary_location_loss(depths, gt, valid)
            + 0.5 * spread_loss(depths, gt, valid))
    if overshoot > 0:
        loss = loss + overshoot * overshoot_loss(depths, gt, valid)
    g = gt.transpose(0, 1).reshape(B * T, 1, *gt.shape[-2:])
    v = valid.transpose(0, 1).reshape(B * T, 1, *valid.shape[-2:])
    if flat > 0:
        mask, flat_valid = flat_mask(g, v)
        if synthetic is not None:
            mask = mask & synthetic
        p = depths.transpose(0, 1).reshape(B * T, 1, *depths.shape[-2:])
        if flat_mode == "plane":
            term = flat_plane_loss(p, g, flat_valid, mask, flat_win)
        elif flat_mode == "band":
            term = flat_band_loss(p, g, flat_valid, mask, flat_win, flat_win_hi)
        else:
            term = flat_excess_loss(p, g, flat_valid, mask)
        loss = loss + flat * term
    return depths, loss + 0.2 * bin_ce_loss(torch.cat(bin_logits, 0), decoder.bin_centres(), g, v)


def train(args):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    arm = Path(args.out) / args.backbone
    if (arm / "decoder.pt").exists():
        raise SystemExit(f"{arm / 'decoder.pt'} exists; use a new directory")
    arm.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    backbone = BACKBONES[args.backbone]().to(dev)
    decoder = build_decoder(backbone.dim).to(dev).train()
    bin_logits = []
    decoder.head.register_forward_hook(lambda module, inputs, output: bin_logits.append(output))
    config = tomllib.loads((ROOT / "configs/main_v8.toml").read_text())
    dataset, sampler = build_mixed(config["data"], clip_len=4, size=256, holdout=config["holdout"],
                                   val=False, tag_source=False, augment=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=sampler,
                                         num_workers=args.workers, drop_last=True,
                                         persistent_workers=args.workers > 0,
                                         prefetch_factor=4 if args.workers > 0 else None,
                                         pin_memory=True)
    params = list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    ema = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    log = open(arm / "train.log", "a")

    def save(path, step):
        torch.save({"decoder": decoder.state_dict(), "ema": ema, "step": step,
                    "meta": {"backbone": args.backbone, "dim": backbone.dim, "taps": TAPS,
                             "args": vars(args), "native_checkpoint": str(NATIVE.relative_to(ROOT)),
                             "mambavision_weights": str(MV_WEIGHTS.relative_to(ROOT))}}, path)

    step, start = 0, time.time()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            clip, gt, valid = (x.to(dev, non_blocking=True) for x in batch[:3])
            B, T = clip.shape[:2]
            # frame-major order, as the native forward_clip feeds its decoder
            frames = clip.transpose(0, 1).reshape(T * B, *clip.shape[2:])
            feats = backbone(frames)
            bin_logits.clear()
            depths, loss = native_depth_loss(decoder, feats, frames, clip, gt, valid, bin_logits,
                                             backbone.size)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            for group in opt.param_groups:
                group["lr"] = lr_at(step, args.lr, args.steps, args.warmup)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ema_update_(ema, decoder.state_dict(), 0.999)
            step += 1
            if step % 50 == 0 or step == 1:
                line = (f"step {step} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} "
                        f"depth_std {depths.std().item():.3f} elapsed {time.time() - start:.0f}s")
                print(line, flush=True)
                log.write(line + "\n"); log.flush()
            if step % 1000 == 0:
                save(arm / "partial.pt", step)
    save(arm / "decoder.pt", step)
    (arm / "partial.pt").unlink(missing_ok=True)
    print("TRAIN_DONE", args.backbone, step, flush=True)


def finetune(args):
    """Joint depth fine-tuning of a probed backbone with its decoder: --source arm -> --backbone arm.

    Starts from the source arm's distilled backbone and EMA decoder, unfreezes both, and
    trains them on the native loss set, data and batch shape used by `train`. The backbone
    learns at --backbone-lr, the decoder at --lr; stage 3 and the classifier stay frozen.
    EMA weights of both are saved.
    """
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    arm = Path(args.out) / args.backbone
    if (arm / "decoder.pt").exists():
        raise SystemExit(f"{arm / 'decoder.pt'} exists; use a new directory")
    arm.mkdir(parents=True, exist_ok=True)
    backbone = BACKBONES[args.source]().cuda()
    decoder = build_decoder(backbone.dim)
    source = torch.load(Path(args.out) / args.source / "decoder.pt", map_location="cpu", weights_only=False)
    decoder.load_state_dict(source["ema"], strict=True)
    decoder = decoder.cuda().train()
    model = backbone.model
    model.requires_grad_(True).train()
    for unused in (model.levels[3], model.norm, model.head, model.levels[2].downsample):
        unused.requires_grad_(False).eval()
    bin_logits = []
    decoder.head.register_forward_hook(lambda module, inputs, output: bin_logits.append(output))
    config = tomllib.loads((ROOT / "configs/main_v8.toml").read_text())
    dataset, sampler = build_mixed(config["data"], clip_len=4, size=256, holdout=config["holdout"],
                                   val=False, tag_source=args.flat_synthetic_only, augment=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=sampler,
                                         num_workers=args.workers, drop_last=True,
                                         persistent_workers=args.workers > 0,
                                         prefetch_factor=4 if args.workers > 0 else None,
                                         pin_memory=True)
    backbone_params = [p for p in model.parameters() if p.requires_grad]
    params = backbone_params + list(decoder.parameters())
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": args.backbone_lr},
                             {"params": list(decoder.parameters()), "lr": args.lr}])
    bases = [args.backbone_lr, args.lr]
    ema_decoder = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    ema_backbone = {k: v.detach().clone() for k, v in model.state_dict().items()}
    log = open(arm / "finetune.log", "a")
    step, start = 0, time.time()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            clip, gt, valid = (x.cuda(non_blocking=True) for x in batch[:3])
            B, T = clip.shape[:2]
            frames = clip.transpose(0, 1).reshape(T * B, *clip.shape[2:])
            feats = backbone.features(frames)
            bin_logits.clear()
            synthetic = (batch[-1].cuda().view(1, B, 1, 1, 1).expand(T, B, 1, 1, 1).reshape(T * B, 1, 1, 1) > 0.5
                         if args.flat_synthetic_only else None)
            depths, loss = native_depth_loss(decoder, feats, frames, clip, gt, valid, bin_logits,
                                             backbone.size, args.boundary_weight, args.overshoot_weight,
                                             args.flat_weight, synthetic, args.flat_mode, args.flat_win,
                                             args.flat_win_hi)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            for group, base in zip(opt.param_groups, bases):
                group["lr"] = lr_at(step, base, args.steps, args.warmup)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ema_update_(ema_decoder, decoder.state_dict(), 0.999)
            ema_update_(ema_backbone, model.state_dict(), 0.999)
            step += 1
            if step % 50 == 0 or step == 1:
                line = (f"step {step} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e}/"
                        f"{opt.param_groups[1]['lr']:.2e} depth_std {depths.std().item():.3f} "
                        f"elapsed {time.time() - start:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
                print(line, flush=True)
                log.write(line + "\n"); log.flush()
    meta = {"source": args.source, "args": vars(args)}
    torch.save({"backbone": ema_backbone, "step": step, "meta": meta}, arm / "backbone.pt")
    torch.save({"decoder": decoder.state_dict(), "ema": ema_decoder, "step": step, "meta": meta},
               arm / "decoder.pt")
    print("FINETUNE_DONE", args.backbone, step, flush=True)


@torch.no_grad()
def predict(args):
    arm = Path(args.out) / args.backbone
    dev = "cuda"
    backbone, decoder, decoder_step = frozen_probe(args.backbone, args.out)
    head = None
    if args.calibrated:
        head = CalibrationHead(backbone.dim)
        head.load_state_dict(torch.load(arm / "calib.pt", map_location="cpu", weights_only=False)["head"], strict=True)
        head = head.to(dev).eval()

    def run(frames):
        feats = backbone(frames)
        depth = depth_at(decoder, feats, frames, backbone.size)
        return depth if head is None else head(depth, feats[-1].mean((2, 3)))

    folder = arm / ("predictions_calibrated" if args.calibrated else "predictions")
    folder.mkdir(exist_ok=True)
    meta = {"step": decoder_step, "weights": "ema", "calibrated": args.calibrated, "sequences": {}}
    for name, (source, start, end) in BEHAVE.items():
        data = torch.load(ROOT / source / f"{name}.pt", map_location="cpu", weights_only=False)
        end = start + min(end - start, args.frames) if args.frames else end
        rgb = data["rgb"][start:end].permute(0, 3, 1, 2).float() / 255
        del data
        chunks = []
        for i in range(0, len(rgb), 32):
            frames = rgb[i:i + 32].to(dev)
            chunks.append(run(frames).float().cpu())
        prediction = torch.cat(chunks)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"non-finite prediction in {name}")
        torch.save(prediction, folder / f"{name}_predictions.pt")
        meta["sequences"][name] = {"start": start, "end": end, "shape": list(prediction.shape)}
        print("predicted", name, tuple(prediction.shape), flush=True)
    frames = rgb[:64].to(dev)
    for i in range(16):
        run(frames[i:i + 1])
    times = []
    for i in range(len(frames)):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        run(frames[i:i + 1])
        torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
    meta["latency_backbone_plus_decoder_ms_median_batch1_fp32_eager"] = statistics.median(times)
    (arm / ("predict_calibrated_meta.json" if args.calibrated else "predict_meta.json")).write_text(json.dumps(meta, indent=2) + "\n")
    print("PREDICT_DONE", args.backbone, meta["latency_backbone_plus_decoder_ms_median_batch1_fp32_eager"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "predict", "calibrate", "finetune"))
    parser.add_argument("--source", default="mambavision_da2", help="finetune: arm to start from")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="finetune: backbone learning rate")
    parser.add_argument("--boundary-weight", type=float, default=3.0, help="finetune: boundary_location_loss weight")
    parser.add_argument("--overshoot-weight", type=float, default=0.0, help="finetune: overshoot_loss weight")
    parser.add_argument("--flat-weight", type=float, default=0.0, help="finetune: GT-flat excess gradient weight")
    parser.add_argument("--flat-synthetic-only", action="store_true", help="finetune: flat term on synthetic-GT frames only")
    parser.add_argument("--flat-mode", choices=("excess", "plane", "band"), default="excess",
                        help="finetune: flat term form -- excess gradient over GT, local plane-fit "
                             "residual, or a win..win-hi pixel bandpass")
    parser.add_argument("--flat-win", type=int, default=9,
                        help="finetune: plane-residual window, or band mode's low (fine) cutoff, in pixels (odd)")
    parser.add_argument("--flat-win-hi", type=int, default=17,
                        help="finetune: band mode's high (coarse) cutoff in pixels (odd, > --flat-win)")
    parser.add_argument("--distilled-hr", default=str(DISTILLED_HR), help="592 px student checkpoint")
    parser.add_argument("--distilled-mr", default=str(DISTILLED_MR), help="384 px student checkpoint")
    parser.add_argument("--backbone", choices=tuple(BACKBONES), required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", default=str(OUT), help="output root (smoke runs go elsewhere)")
    parser.add_argument("--calibrated", action="store_true", help="predict through calib.pt")
    parser.add_argument("--frames", type=int, default=0, help="limit frames per sequence (smoke only)")
    args = parser.parse_args()
    FineTunedBackbone.student = Path(args.out) / "mambavision_da2_ft" / "backbone.pt"
    FineTunedHighResBackbone.student = Path(args.out) / "mambavision_da2_hr_ft" / "backbone.pt"
    HighResDistilledBackbone.student = Path(args.distilled_hr)
    MidResDistilledBackbone.student = Path(args.distilled_mr)
    FineTunedMidResBackbone.student = Path(args.out) / "mambavision_da2_mr_ft" / "backbone.pt"
    {"train": train, "predict": predict, "calibrate": calibrate, "finetune": finetune}[args.mode](args)


if __name__ == "__main__":
    main()
