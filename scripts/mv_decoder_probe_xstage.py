"""Cross-stage tap ablation: continues appendix M's "how many/which backbone taps feed
the decoder" question past stage 2's own block count (8, appendix M's ceiling). MambaVision-T
has 16 blocks total across all 4 stages (1 + 3 + 8 + 4) at four different (channel,
resolution) pairs, traced directly from the loaded model:

  stage 0: 1 block,  80 ch @ 64x64
  stage 1: 3 blocks, 160 ch @ 32x32
  stage 2: 8 blocks, 320 ch @ 16x16  (the grid appendix F/M/J/K already tap)
  stage 3: 4 blocks, 640 ch @  8x8   (no downsample after)

--taps 12 = stage 2 + stage 3 only (two resolutions). --taps 16 = every block in the
network (all four).

New file, new output root (work_dirs/mv_decoder_probe_xstage_20260917/) -- only imports
from scripts/mv_decoder_probe.py and scripts/mv_decoder_probe_bigdecoder.py, writes to
neither of their trees, so appendices A-M stay exactly reproducible.

DPTDecoder itself is untouched. It fuses taps as `sum(p(f) for p, f in
zip(self.proj, feats2d))`, which needs every f the same (16x16) spatial size AND every p
matched to f's channel count. Appendix M's `set_n_blocks` made all proj layers identical
because every M/J/K/L tap was 320-channel/16x16. That no longer holds here, so this file
builds `decoder.proj` by hand (one Conv2d per tap, its own in_channels) and bilinear-
resizes non-16x16 taps to 16x16 inside the backbone's features() -- before the decoder
ever sees them, so DPTDecoder.forward runs unmodified.

  python scripts/mv_decoder_probe_xstage.py train    --taps 16 --steps 4000
  python scripts/mv_decoder_probe_xstage.py finetune --taps 16 --steps 4000
  python scripts/mv_decoder_probe_xstage.py calibrate --taps 16 --steps 4000
  python scripts/mv_decoder_probe_xstage.py predict --taps 16 --calibrated
  python scripts/mv_decoder_probe_xstage.py score --taps 16
"""
import argparse
import random
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sokkanaem import from_checkpoint
from sokkanaem.data import build_mixed
from sokkanaem.ema import ema_update_
from sokkanaem.model import DPTDecoder
from sokkanaem.schedule import lr_at
from scripts.mv_decoder_probe import (
    BEHAVE, DISTILLED_HR, NATIVE, CalibrationHead, HighResDistilledBackbone,
    depth_at, native_depth_loss)
# score_sequence/gate need cv2 (sokkanaem env), imported lazily inside score() -- the
# other modes run in the mambavision env, which doesn't have it.

OUT = ROOT / "work_dirs/mv_decoder_probe_xstage_20260917"
ORIG_SCORE = ROOT / "work_dirs/mv_decoder_probe_20260915/score.json"
# the grid DPTDecoder's proj+sum expects. NOT a fixed 16: F/M/G1 all run the 592px
# distilled backbone, whose own stage-2 grid is 592 // 16 = 37 (the decoder's fusion
# pyramid then doubles 3x to reach the stem skips' 74/148/296, which are themselves
# 1/8, 1/4, 1/2 of the 592px "wide" frame depth_at upsamples to -- an all-16x16 taps
# list would size-mismatch every skip connection, exactly like stage 2's own untouched
# tap already has to sit at 37, not 16, in the existing pipeline).
FUSION_SIZE = HighResDistilledBackbone.size // 16

# (level, block, channels) for every block in the network, in forward order.
ALL_TAPS = ([(0, 0, 80)]
            + [(1, i, 160) for i in range(3)]
            + [(2, i, 320) for i in range(8)]
            + [(3, i, 640) for i in range(4)])
TAP_PRESETS = {12: [t for t in ALL_TAPS if t[0] in (2, 3)], 16: ALL_TAPS}

ARM = "mambavision_da2_hr_16taps"    # stage-1 (train) name: taps+seed only, never tagged
FT_ARM = ARM                        # stage-2-onward name: ARM + --tag, for shape-loss combos
TAPS = TAP_PRESETS[16]


class CrossStageBackbone(HighResDistilledBackbone):
    """HighResDistilledBackbone, but taps blocks from any of the 4 stages (module global
    TAPS, a (level, block_index, channels) list set from --taps in main()), not just
    stage 2. Stage 0/1 blocks are plain ConvBlocks (B,C,H,W) already, no windowing;
    stage 2/3 use a single global window each (window_size == their full grid), the same
    reshape appendix M's AllTapsBackbone already used for stage 2."""
    student = DISTILLED_HR

    def __init__(self):
        super().__init__()
        # level3's window_size is hardcoded to 8 in MambaVisionBackbone.__init__
        # (window_size=[8, 8, size // 16, 8]) regardless of `size` -- fine for the
        # existing pipeline, which never runs level3 at all. At 592px, level3's actual
        # grid is 19x19 (patch_embed /4, three more /2 downsamples: 592/4=148 ->74->37->19),
        # not 8x8, so the single-global-window assumption features() makes for stage 2/3
        # needs level3.window_size widened to match -- exactly what level2's own
        # `size // 16` entry already does. Attention (mamba_vision.py Attention class) has
        # no relative position bias or other window-size-shaped weight, only a runtime
        # partition/reverse, so this is a free, weight-compatible override -- traced
        # dynamically rather than hardcoded so it still works if `size` ever changes.
        # level2's Mamba mixer only has a CUDA kernel (mamba_ssm's selective_scan_cuda),
        # so the probe forward has to run on GPU even though __init__ hasn't moved the
        # real model there yet.
        with torch.no_grad():
            self.cuda()
            probe = torch.zeros(1, 3, self.size, self.size, device="cuda")
            m = self.model
            x = m.patch_embed((probe - self.mean) / self.std)
            x = m.levels[0](x)
            x = m.levels[1](x)
            x = m.levels[2](x)
            self.cpu()
        grid = x.shape[-1]
        if grid != m.levels[3].window_size:
            m.levels[3].window_size = grid

    def features(self, frames):
        m = self.model
        if frames.shape[-1] != self.size:
            frames = F.interpolate(frames, size=(self.size, self.size), mode="bilinear",
                                   align_corners=False)
        wanted = {(lvl, i) for lvl, i, _ in TAPS}
        x = m.patch_embed((frames - self.mean) / self.std)
        feats = []
        for lvl in (0, 1):
            level = m.levels[lvl]
            for i, block in enumerate(level.blocks):
                x = block(x)
                if (lvl, i) in wanted:
                    feats.append(x)
            x = level.downsample(x)
        for lvl in (2, 3):
            level = m.levels[lvl]
            n, c, h, w = x.shape
            if not h == w == level.window_size:
                raise ValueError(f"level {lvl} expects one global window, got {h}x{w} "
                                 f"vs window_size {level.window_size}")
            xw = self.window_partition(x, level.window_size)
            for i, block in enumerate(level.blocks):
                xw = block(xw)
                if (lvl, i) in wanted:
                    feats.append(xw.transpose(1, 2).reshape(n, c, h, w))
            x = xw.transpose(1, 2).reshape(n, c, h, w)
            if level.downsample is not None:
                x = level.downsample(x)
        assert len(feats) == len(TAPS), (len(feats), len(TAPS))
        return [f if f.shape[-1] == FUSION_SIZE else
                F.interpolate(f, size=(FUSION_SIZE, FUSION_SIZE), mode="bilinear",
                             align_corners=False)
                for f in feats]


def build_xstage_decoder(seed=0):
    """Same native-weight transplant appendix F/M used (everything but `proj.*` is
    key-for-key native structure), but `proj` is built by hand: one Conv2d per tap, its
    own in_channels, since taps here span 4 different channel counts."""
    reference = from_checkpoint(NATIVE, "cpu").decoder
    torch.manual_seed(seed)
    decoder = DPTDecoder(TAPS[0][2], patch_size=16, width=reference.width, bins=reference.bins,
                         fuse_norm=reference.norm_x is not None, full_res=reference.full_res)
    decoder.proj = torch.nn.ModuleList(
        torch.nn.Conv2d(chans, decoder.width, 1) for _, _, chans in TAPS)
    state = {k: v for k, v in reference.state_dict().items() if not k.startswith("proj.")}
    missing, unexpected = decoder.load_state_dict(state, strict=False)
    if unexpected or not all(k.startswith("proj.") for k in missing):
        raise RuntimeError(f"decoder transplant mismatch: {missing} {unexpected}")
    return decoder


def train(args):
    """Stage 1: decoder only, backbone frozen. Mirrors mv_decoder_probe.train()."""
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    arm = OUT / ARM
    if (arm / "decoder.pt").exists():
        raise SystemExit(f"{arm / 'decoder.pt'} exists; refusing to overwrite")
    arm.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    backbone = CrossStageBackbone().to(dev)
    decoder = build_xstage_decoder(args.seed).to(dev).train()
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
    step, start = 0, time.time()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            clip, gt, valid = (x.to(dev, non_blocking=True) for x in batch[:3])
            B, T = clip.shape[:2]
            frames = clip.transpose(0, 1).reshape(T * B, *clip.shape[2:])
            feats = backbone(frames)
            bin_logits.clear()
            depths, loss = native_depth_loss(decoder, feats, frames, clip, gt, valid, bin_logits,
                                             backbone.size)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            for group in opt.param_groups:
                group["lr"] = lr_at(step, args.lr, args.steps, args.warmup)
            opt.zero_grad(set_to_none=True)
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
    torch.save({"decoder": decoder.state_dict(), "ema": ema, "step": step,
                "meta": {"backbone": ARM, "taps": TAPS, "args": vars(args)}}, arm / "decoder.pt")
    print("TRAIN_DONE", ARM, step, flush=True)


def finetune(args):
    """Stage 2: unfreeze backbone too, joint train from stage 1's decoder."""
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    arm = OUT / (FT_ARM + "_ft")
    if (arm / "decoder.pt").exists():
        raise SystemExit(f"{arm / 'decoder.pt'} exists; refusing to overwrite")
    arm.mkdir(parents=True, exist_ok=True)
    backbone = CrossStageBackbone().cuda()
    decoder = build_xstage_decoder(args.seed)
    source = torch.load(OUT / ARM / "decoder.pt", map_location="cpu", weights_only=False)
    decoder.load_state_dict(source["ema"], strict=True)
    decoder = decoder.cuda().train()
    model = backbone.model
    model.requires_grad_(True).train()
    # unlike appendix F/M (stem..stage2 only, stage3 dead weight), the forward pass here
    # always runs through every stage to reach stage3's taps (even the 12-taps preset,
    # which just doesn't collect stage0/1) -- only the classification head is unused.
    for unused in (model.norm, model.head):
        unused.requires_grad_(False).eval()
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
            depths, loss = native_depth_loss(decoder, feats, frames, clip, gt, valid, bin_logits,
                                             backbone.size, args.boundary_weight, args.overshoot_weight,
                                             args.flat_weight, None, args.flat_mode, args.flat_win,
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
                        f"elapsed {time.time() - start:.0f}s")
                print(line, flush=True)
                log.write(line + "\n"); log.flush()
    meta = {"source": ARM, "args": vars(args)}
    torch.save({"backbone": ema_backbone, "step": step, "meta": meta}, arm / "backbone.pt")
    torch.save({"decoder": decoder.state_dict(), "ema": ema_decoder, "step": step, "meta": meta},
               arm / "decoder.pt")
    print("FINETUNE_DONE", FT_ARM + "_ft", step, flush=True)


class FineTunedCrossStageBackbone(CrossStageBackbone):
    student = None   # set from --out in main()


def frozen_probe(backbone_cls, decoder_path):
    checkpoint = torch.load(decoder_path, map_location="cpu", weights_only=False)
    backbone = backbone_cls().cuda()
    decoder = build_xstage_decoder()
    decoder.load_state_dict(checkpoint["ema"], strict=True)
    return backbone, decoder.cuda().eval().requires_grad_(False), checkpoint["step"]


def calibrate(args):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    arm = OUT / (FT_ARM + "_ft")
    if (arm / "calib.pt").exists():
        raise SystemExit(f"{arm / 'calib.pt'} exists; refusing to overwrite")
    FineTunedCrossStageBackbone.student = arm / "backbone.pt"
    backbone, decoder, decoder_step = frozen_probe(FineTunedCrossStageBackbone, arm / "decoder.pt")
    head = CalibrationHead(TAPS[-1][2]).cuda().train()   # feats[-1]'s channel count
    config = tomllib.loads((ROOT / "configs/main_v8.toml").read_text())
    dataset, sampler = build_mixed(config["data"][:2], clip_len=4, size=256,
                                   holdout=config["holdout"], augment=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler,
                                         num_workers=args.workers, drop_last=True,
                                         persistent_workers=args.workers > 0, pin_memory=True)
    from sokkanaem.losses import si_log_loss
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
            for group in opt.param_groups:
                group["lr"] = 1e-3 * (step + 1) / args.steps
            opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                line = (f"step {step} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} "
                        f"median {out.median().item():.2f} m elapsed {time.time() - start:.0f}s")
                print(line, flush=True)
                log.write(line + "\n"); log.flush()
    torch.save({"head": head.state_dict(), "step": step, "decoder_step": decoder_step,
                "meta": {"backbone": FT_ARM + "_ft", "args": vars(args)}}, arm / "calib.pt")
    print("CALIB_DONE", FT_ARM + "_ft", step, flush=True)


@torch.no_grad()
def predict(args):
    arm = OUT / (FT_ARM + "_ft")
    dev = "cuda"
    FineTunedCrossStageBackbone.student = arm / "backbone.pt"
    backbone, decoder, decoder_step = frozen_probe(FineTunedCrossStageBackbone, arm / "decoder.pt")
    head = None
    if args.calibrated:
        head = CalibrationHead(TAPS[-1][2])
        head.load_state_dict(torch.load(arm / "calib.pt", map_location="cpu", weights_only=False)["head"],
                             strict=True)
        head = head.to(dev).eval()

    def run(frames):
        feats = backbone(frames)
        depth = depth_at(decoder, feats, frames, backbone.size)
        return depth if head is None else head(depth, feats[-1].mean((2, 3)))

    folder = arm / ("predictions_calibrated" if args.calibrated else "predictions")
    folder.mkdir(exist_ok=True)
    for name, (source, start, end) in BEHAVE.items():
        data = torch.load(ROOT / source / f"{name}.pt", map_location="cpu", weights_only=False)
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
        print("predicted", name, tuple(prediction.shape), flush=True)
    print("PREDICT_DONE", FT_ARM + "_ft", flush=True)


def score(args):
    """Score the calibrated predictions against GT, then print ratios against the
    untouched F (4-taps), G1, and appendix M's 8-taps numbers."""
    import json
    from scripts.evaluate_behave_pilot import score_sequence
    from scripts.evaluate_behave_localrisk import gate
    orig = json.loads(ORIG_SCORE.read_text())["combined"]
    ref_f = orig["probe_mambavision_da2_hr_ft_calibrated"]["metrics"]
    ref_g1 = orig["shape_g1_calibrated"]["metrics"]
    q0 = orig["q0_dense"]["metrics"]
    m8_path = ROOT / "work_dirs/mv_decoder_probe_bigdecoder_20260917/mambavision_da2_hr_8taps_ft/score.json"
    ref_m8 = json.loads(m8_path.read_text())["combined"] if m8_path.exists() else None
    keys = ("absrel_raw", "absrel_edge", "overshoot", "boundary_f1", "flat_tv")
    arm = OUT / (FT_ARM + "_ft")
    per_seq = {}
    for name, (source, start, end) in BEHAVE.items():
        data = torch.load(ROOT / source / f"{name}.pt", map_location="cpu", weights_only=False)
        gt = data["depth_mm"][start:end, None].float() / 1000
        del data
        prediction = torch.load(arm / "predictions_calibrated" / f"{name}_predictions.pt",
                                map_location="cpu")
        metrics, _ = score_sequence(prediction, gt)
        per_seq[name] = {k: metrics[k] for k in keys}
        print("scored", name, flush=True)
    combined = {k: float(np.mean([per_seq[s][k] for s in per_seq])) for k in keys}
    passing = sum(gate(q0, per_seq[s])["pass"] for s in per_seq)
    print(f"\n{FT_ARM} F-equivalent, combined metrics:", combined)
    print(f"Q0 gate: {passing}/6")
    print("ratio to F (4 taps):     ", {k: round(combined[k] / ref_f[k], 3) for k in keys})
    print("ratio to G1 (best, 592): ", {k: round(combined[k] / ref_g1[k], 3) for k in keys})
    if ref_m8:
        print("ratio to M (8 taps):     ", {k: round(combined[k] / ref_m8[k], 3) for k in keys})
    print("ratio to Q0:             ", {k: round(combined[k] / q0[k], 3) for k in keys})
    (arm / "score.json").write_text(json.dumps(
        {"per_sequence": per_seq, "combined": combined, "sequences_passing_q0_gate": passing},
        indent=2) + "\n")
    print("SCORE_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "finetune", "calibrate", "predict", "score"))
    ap.add_argument("--taps", type=int, choices=(12, 16), default=16)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--calibrated", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="train/finetune seed; seed 0 is "
                     "the original arm's directory name (unsuffixed), seed>0 gets its own "
                     "directory so it never collides with or overwrites the seed-0 result")
    ap.add_argument("--boundary-weight", type=float, default=3.0, help="finetune: "
                     "boundary_location_loss weight (appendix N's plain arms use the default)")
    ap.add_argument("--overshoot-weight", type=float, default=0.0, help="finetune: overshoot_loss weight")
    ap.add_argument("--flat-weight", type=float, default=0.0, help="finetune: GT-flat excess/band weight")
    ap.add_argument("--flat-mode", choices=("excess", "plane", "band"), default="excess")
    ap.add_argument("--flat-win", type=int, default=9, help="plane window, or band mode's low cutoff")
    ap.add_argument("--flat-win-hi", type=int, default=17, help="band mode's high cutoff")
    ap.add_argument("--tag", default="", help="extra suffix for the arm directory, so a "
                     "shape-loss combo (e.g. G1's band recipe on this backbone) never "
                     "collides with appendix N's plain-recipe arm of the same --taps/--seed")
    args = ap.parse_args()
    global ARM, FT_ARM, TAPS
    TAPS = TAP_PRESETS[args.taps]
    ARM = f"mambavision_da2_hr_{args.taps}taps" + (f"_s{args.seed}" if args.seed else "")
    FT_ARM = ARM + (f"_{args.tag}" if args.tag else "")
    {"train": train, "finetune": finetune, "calibrate": calibrate, "predict": predict,
     "score": score}[args.mode](args)


if __name__ == "__main__":
    main()
