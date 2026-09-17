"""Decoder-capacity ablation: does feeding the fusion decoder ALL 8 stage-2 taps
(instead of the 4 scripts/mv_decoder_probe.py uses) close any of the F1/overshoot/flat-TV
gap that loss-term tuning (SOKKANAEM_MV_DECODER_PROBE_20260915.md appendices J/K/L) could
not? Everything here is new: this file, and its output root
work_dirs/mv_decoder_probe_bigdecoder_20260917/. scripts/mv_decoder_probe.py and its
work_dirs/mv_decoder_probe_20260915/ tree are only imported from / read from, never
written to, so every existing appendix stays exactly reproducible.

Same 592px distilled backbone weights (work_dirs/mv_da2_distill_hr_20260915/student.pt,
frozen at this stage) and the same two-stage recipe appendix F used (train: decoder only,
4000 step; finetune: decoder+backbone joint, 4000 step; boundary=3, no overshoot/flat
term) -- so the only difference from the existing F arm (OUT_ORIG/mambavision_da2_hr_ft)
is tap count. `set_n_blocks` already parametrizes how many per-block 1x1 proj layers the
decoder owns, so this needed zero sokkanaem/model.py changes.

  python scripts/mv_decoder_probe_bigdecoder.py train    --steps 4000
  python scripts/mv_decoder_probe_bigdecoder.py finetune --steps 4000
  python scripts/mv_decoder_probe_bigdecoder.py calibrate --steps 4000
  python scripts/mv_decoder_probe_bigdecoder.py predict --calibrated
  python scripts/mv_decoder_probe_bigdecoder.py score
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
# score_sequence/gate need cv2 (scripts.evaluate_behave_pilot), which lives in the
# `sokkanaem` env, not the `mambavision` env train/finetune/calibrate/predict run under -
# imported lazily inside score() so the other modes don't need it.

OUT = ROOT / "work_dirs/mv_decoder_probe_bigdecoder_20260917"
# ARM/N_TAPS/TAP_INDICES are set from --taps in main() before any mode function runs, so a
# 6-taps sweep point can reuse this same file/root as the original all-8 ablation (appendix
# M) without duplicating it. --taps 8 (the default) reproduces appendix M's arm exactly.
ARM = "mambavision_da2_hr_8taps"    # stage-1 (train) name: taps+seed only, never tagged
FT_ARM = ARM                        # stage-2-onward name: ARM + --tag, for shape-loss combos
N_TAPS = 8
TAP_INDICES = tuple(range(8))
# the reference ratios this ablation is judged against: appendix F/G1's own numbers,
# read once from the untouched existing score.json (never written to).
ORIG_SCORE = ROOT / "work_dirs/mv_decoder_probe_20260915/score.json"


def spaced_indices(n, total=8):
    """n evenly spaced block indices out of `total` (0-indexed, both ends included when
    n>1). n=8 -> all of them (reproduces appendix M); n=4 is NOT scripts/mv_decoder_probe.py's
    TAPS=(1,3,5,7) -- that arm (F) already exists and isn't reproduced here."""
    if n >= total:
        return tuple(range(total))
    return tuple(sorted({round(i * (total - 1) / (n - 1)) for i in range(n)}))


class AllTapsBackbone(HighResDistilledBackbone):
    """HighResDistilledBackbone, but returns the TAP_INDICES subset of stage-2 block
    tokens (module global, set from --taps in main()) instead of the TAPS=(1,3,5,7)
    scripts/mv_decoder_probe.py hardcodes. Despite the name, taps need not be all 8 --
    kept for appendix M's naming continuity."""
    student = DISTILLED_HR

    def features(self, frames):
        m = self.model
        if frames.shape[-1] != self.size:
            frames = F.interpolate(frames, size=(self.size, self.size), mode="bilinear",
                                   align_corners=False)
        x = m.levels[1](m.levels[0](m.patch_embed((frames - self.mean) / self.std)))
        level = m.levels[2]
        n, c, h, w = x.shape
        if not h == w == level.window_size:
            raise ValueError("expects one global stage-2 window, i.e. 256 px input")
        x = self.window_partition(x, level.window_size)
        feats = []
        for i, block in enumerate(level.blocks):
            x = block(x)
            if i in TAP_INDICES:
                feats.append(x.transpose(1, 2).reshape(n, c, h, w))
        assert len(feats) == N_TAPS
        return feats


def build_wide_decoder(dim, n_taps, seed=0):
    """Same transplant scripts/mv_decoder_probe.py's build_decoder does, just with
    set_n_blocks(n_taps) instead of the hardcoded 4 -- every other weight (stem, reduce,
    mix, head, up_* ) is untouched native structure, so the transplant still matches
    key-for-key outside `proj.*`."""
    reference = from_checkpoint(NATIVE, "cpu").decoder
    torch.manual_seed(seed)
    decoder = DPTDecoder(dim, patch_size=16, width=reference.width, bins=reference.bins,
                         fuse_norm=reference.norm_x is not None, full_res=reference.full_res)
    decoder.set_n_blocks(n_taps)
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
    backbone = AllTapsBackbone().to(dev)
    decoder = build_wide_decoder(backbone.dim, N_TAPS, args.seed).to(dev).train()
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
                "meta": {"backbone": ARM, "dim": backbone.dim, "n_taps": N_TAPS, "args": vars(args)}},
               arm / "decoder.pt")
    print("TRAIN_DONE", ARM, step, flush=True)


def finetune(args):
    """Stage 2: unfreeze backbone too, joint train from stage 1's decoder.

    Resumable: this environment has been killing long background runs outright (not just
    IPC/terminal loss -- full sandbox restarts, which setsid/nohup/disown cannot survive)
    roughly every 30-40 minutes, so an 8000-step run has repeatedly died mid-flight with
    nothing saved (torch.save only ran at the very end). `partial.pt` now checkpoints
    decoder+backbone+optimizer+step every 500 steps; a restart with the same --tag/--taps
    picks it back up instead of losing the whole run.
    """
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    arm = OUT / (FT_ARM + "_ft")
    if (arm / "decoder.pt").exists():
        raise SystemExit(f"{arm / 'decoder.pt'} exists; refusing to overwrite")
    arm.mkdir(parents=True, exist_ok=True)
    backbone = AllTapsBackbone().cuda()
    decoder = build_wide_decoder(backbone.dim, N_TAPS, args.seed)
    source = torch.load(OUT / ARM / "decoder.pt", map_location="cpu", weights_only=False)
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
    step, start = 0, time.time()
    partial_path = arm / "partial.pt"
    if partial_path.exists():
        ckpt = torch.load(partial_path, map_location="cuda", weights_only=False)
        decoder.load_state_dict(ckpt["decoder"]); model.load_state_dict(ckpt["backbone"])
        opt.load_state_dict(ckpt["opt"]); ema_decoder = ckpt["ema_decoder"]
        ema_backbone = ckpt["ema_backbone"]; step = ckpt["step"]
        print(f"RESUMED from partial.pt at step {step}", flush=True)
    log = open(arm / "finetune.log", "a")
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
            if step % 500 == 0:
                torch.save({"decoder": decoder.state_dict(), "backbone": model.state_dict(),
                            "opt": opt.state_dict(), "ema_decoder": ema_decoder,
                            "ema_backbone": ema_backbone, "step": step}, partial_path)
    meta = {"source": ARM, "args": vars(args)}
    torch.save({"backbone": ema_backbone, "step": step, "meta": meta}, arm / "backbone.pt")
    torch.save({"decoder": decoder.state_dict(), "ema": ema_decoder, "step": step, "meta": meta},
               arm / "decoder.pt")
    partial_path.unlink(missing_ok=True)
    print("FINETUNE_DONE", FT_ARM + "_ft", step, flush=True)


class FineTunedAllTapsBackbone(AllTapsBackbone):
    student = None   # set from --out in main()


def frozen_probe(backbone_cls, decoder_path):
    checkpoint = torch.load(decoder_path, map_location="cpu", weights_only=False)
    backbone = backbone_cls().cuda()
    decoder = build_wide_decoder(backbone.dim, N_TAPS)
    decoder.load_state_dict(checkpoint["ema"], strict=True)
    return backbone, decoder.cuda().eval().requires_grad_(False), checkpoint["step"]


def calibrate(args):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    arm = OUT / (FT_ARM + "_ft")
    if (arm / "calib.pt").exists():
        raise SystemExit(f"{arm / 'calib.pt'} exists; refusing to overwrite")
    FineTunedAllTapsBackbone.student = arm / "backbone.pt"
    backbone, decoder, decoder_step = frozen_probe(FineTunedAllTapsBackbone, arm / "decoder.pt")
    head = CalibrationHead(backbone.dim).cuda().train()
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
    FineTunedAllTapsBackbone.student = arm / "backbone.pt"
    backbone, decoder, decoder_step = frozen_probe(FineTunedAllTapsBackbone, arm / "decoder.pt")
    head = None
    if args.calibrated:
        head = CalibrationHead(backbone.dim)
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
    """Score the calibrated 8-taps predictions against GT, then print ratios against the
    untouched F (4-taps) and G1 numbers already recorded in the original score.json."""
    import json
    from scripts.evaluate_behave_pilot import score_sequence
    from scripts.evaluate_behave_localrisk import gate
    orig = json.loads(ORIG_SCORE.read_text())["combined"]
    ref_f = orig["probe_mambavision_da2_hr_ft_calibrated"]["metrics"]
    ref_g1 = orig["shape_g1_calibrated"]["metrics"]
    q0 = orig["q0_dense"]["metrics"]
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
    print("ratio to Q0:             ", {k: round(combined[k] / q0[k], 3) for k in keys})
    (arm / "score.json").write_text(json.dumps(
        {"per_sequence": per_seq, "combined": combined, "sequences_passing_q0_gate": passing},
        indent=2) + "\n")
    print("SCORE_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "finetune", "calibrate", "predict", "score"))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--calibrated", action="store_true")
    ap.add_argument("--taps", type=int, default=8, help="number of evenly spaced stage-2 "
                     "blocks to tap (of 8 total); 8 reproduces appendix M's arm")
    ap.add_argument("--seed", type=int, default=0, help="train/finetune seed; seed 0 is "
                     "the original arm's directory name (unsuffixed), seed>0 gets its own "
                     "directory so it never collides with or overwrites the seed-0 result")
    ap.add_argument("--boundary-weight", type=float, default=3.0, help="finetune: "
                     "boundary_location_loss weight (appendix M/N's plain arms use the default)")
    ap.add_argument("--overshoot-weight", type=float, default=0.0, help="finetune: overshoot_loss weight")
    ap.add_argument("--flat-weight", type=float, default=0.0, help="finetune: GT-flat excess/band weight")
    ap.add_argument("--flat-mode", choices=("excess", "plane", "band"), default="excess")
    ap.add_argument("--flat-win", type=int, default=9, help="plane window, or band mode's low cutoff")
    ap.add_argument("--flat-win-hi", type=int, default=17, help="band mode's high cutoff")
    ap.add_argument("--tag", default="", help="extra suffix for the arm directory, so a "
                     "shape-loss combo never collides with appendix M/N's plain-recipe arm "
                     "of the same --taps/--seed")
    args = ap.parse_args()
    global ARM, FT_ARM, N_TAPS, TAP_INDICES
    N_TAPS = args.taps
    TAP_INDICES = spaced_indices(args.taps)
    ARM = f"mambavision_da2_hr_{args.taps}taps" + (f"_s{args.seed}" if args.seed else "")
    FT_ARM = ARM + (f"_{args.tag}" if args.tag else "")
    {"train": train, "finetune": finetune, "calibrate": calibrate, "predict": predict,
     "score": score}[args.mode](args)


if __name__ == "__main__":
    main()
