"""The ACC comparison table: one script, every model, one protocol.

PLAN_ACC A2-A4. Ours and the Hugging Face baselines used to be scored by two
different files that agreed by inspection -- and had already drifted once, on
t-delta (REPORT §4.10). Here they share the manifest (A1), the alignment
(sokkanaem/alignment.py, A2), the scorer (sokkanaem/metrics.py) and the
statistics (A3), so the only thing that differs between two rows is the model.

Report separate metric (no GT fit), scale-aligned and relative-shape panels.
Different fitted gauges answer different questions; their scores must not be
combined into one accuracy ranking. See paper/PROTOCOL.md.

Statistics per A3: mean AND median AND 10% trimmed AND the P90/P95 tail, with
the failures counted rather than averaged in. A model whose mean is carried by
a handful of blown-up clips is a different object from one that is uniformly
worse, and the mean alone cannot tell them apart.

    python scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
        --model ours:work_dirs/v11-longclip-spread-s0/latest.pt
    python scripts/eval_acc.py --manifest manifests/acc_real_L8.json \
        --model hf:Intel/dpt-large --space disparity --infer-size 256
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from sokkanaem.alignment import LEGACY_MODES, MODES, align
from sokkanaem.data import load_manifest
from sokkanaem.metrics import boot_ci, clip_scores, pooled, robust
from sokkanaem.sharpness import sharpness_scores
from sokkanaem.protocol import VERSION, check_final_test, provenance, runtime_versions

OUT = Path("work_dirs/acc")
# Fixed relative-shape gauge for the sharpness suite; edge AbsRel depends on it.
SHARP_GAUGE = "scaleshift"

# A6's regions, in report order. Every one is computed from GT depth and RGB
# alone -- never from a model's own output -- so the same pixel belongs to the
# same region in every row of the table.
REGIONS = ("all", "edge", "flat", "dynamic", "static", "near", "mid", "far")


@torch.no_grad()
def region_masks(frames, gt, valid, flow_px=1.5, edge_thr=0.10,
                 bands=(2.0, 4.0)):
    """Per-pixel regions for the A6 failure table. Returns {name: bool mask}.

    edge     within 2 px of a GT depth discontinuity -- a step of `edge_thr`
             in log depth across one pixel, measured only where BOTH sides are
             valid (a Kinect dropout reads 0 m, and an unguarded gradient would
             mark the rim of every hole as a depth edge).
    flat     the valid complement of `edge`.
    dynamic  optical flow that disagrees with the frame's dominant motion by
             more than `flow_px`. Independently moving objects, without
             assuming the camera is still: the median flow absorbs ego-motion.
             This is the region Bonn exists to test and the one §6.4 puts 3.5x
             above its ceiling.
    near/mid/far  fixed metric bands, not per-clip terciles: these are indoor
             Kinect sequences and a fixed band means the same thing in every
             clip, which a quantile does not.
    """
    import torch.nn.functional as F
    from sokkanaem.metrics import _flow

    v = valid.bool()
    ld = torch.log(gt.clamp(min=1e-3))
    g = torch.zeros_like(ld)
    for d in (-1, -2):                      # x then y
        a, b = ld.diff(dim=d).abs(), v.narrow(d, 0, gt.shape[d] - 1) & \
            v.narrow(d, 1, gt.shape[d] - 1)
        step = torch.zeros_like(ld)
        step.narrow(d, 1, gt.shape[d] - 1).copy_(a * b)
        g = g + step
    edge = (F.max_pool2d(((g > edge_thr) & v).float(), 5, 1, 2) > 0) & v

    fl = _flow(frames[1:], frames[:-1])                    # (T-1,2,H,W)
    med = fl.flatten(2).median(-1).values[..., None, None]  # dominant motion
    res = (fl - med).pow(2).sum(1, keepdim=True).sqrt()
    dyn = torch.cat([res[:1], res]) > flow_px               # frame 0 borrows 1
    dyn = dyn & v

    lo, hi = bands
    return {"all": v, "edge": edge, "flat": v & ~edge,
            "dynamic": dyn, "static": v & ~dyn,
            "near": v & (gt < lo), "mid": v & (gt >= lo) & (gt < hi),
            "far": v & (gt >= hi)}


def region_scores(depth, gt, masks):
    """Pixel sums per region, so the dataset-level number is pixel-pooled the
    way the rest of the table is rather than averaged over clips."""
    gc = gt.clamp(min=1e-6)
    rel = (depth - gt).abs() / gc
    d1 = (torch.maximum(depth / gc, gt / depth.clamp(min=1e-6)) < 1.25).float()
    out = {}
    for name, m in masks.items():
        n = float(m.sum().item())
        out[name] = {"rel": float((rel * m).sum().item()),
                     "d1": float((d1 * m).sum().item()), "px": n}
    return out


def ours_runner(ckpt, dev, tau=None, reset_every=0, bypass_temporal=False,
                **kw):
    """Streaming over the whole clip, exactly as deployment would -- the state
    and the change mask are warmed up rather than reset per frame.

    The three overrides are PLAN_ACC A5's decomposition. The reported error is
    one number covering two mechanisms, and A5 asks which of them owns it:

        tau=0           every patch active. Removes the gating, keeps the
                        temporal state. What sparsity costs.
        reset_every=1   fresh state each frame. Removes the temporal state,
                        keeps the gating (which then fires on every frame as a
                        keyframe). What the spatial predictor scores alone.
        bypass_temporal the TemporalBlocks contribute nothing at all, so not
                        even the within-frame readout survives. Separates
                        "state carried across frames" from "the block exists".
    """
    from sokkanaem import from_checkpoint
    model = from_checkpoint(ckpt, dev, **kw).eval()
    if tau is not None:
        model.detector.tau_on, model.detector.tau_off = tau, tau / 2
    if bypass_temporal:
        from sokkanaem.model import TemporalBlock
        for b in model.blocks:
            if isinstance(b, TemporalBlock):
                b.step = lambda tokens, mask, h, gate_mode="delta": (tokens, h)

    eff = {}

    @torch.no_grad()
    def run(clip):
        eff.setdefault("size", (getattr(model, "infer_size", None) or clip.shape[-2],
                                getattr(model, "infer_size", None) or clip.shape[-1]))
        clip = clip[None].to(dev)
        if not reset_every:
            depths, masks = model.forward_clip(clip)
            return depths[0], {"active": masks[:, 1:].mean().item()
                              if masks.shape[1] > 1 else masks.mean().item(),
                              "active_all": masks.mean().item()}
        state, depths, masks = None, [], []
        for t in range(clip.shape[1]):
            if t % reset_every == 0:
                state = None
            d, state, info = model.step(clip[:, t], state)
            depths.append(d)
            masks.append(info["mask"])
        masks = torch.stack(masks, 1)
        return (torch.stack(depths, 1)[0],
                {"active": masks[:, 1:].mean().item() if masks.shape[1] > 1
                           else masks.mean().item(), "active_all": masks.mean().item()})

    return run, "depth", model, eff


def da3_runner(name, dev, chunk=32):
    """Depth Anything 3, which is not a Hugging Face AutoModel and needs the
    `baselines` conda env (`conda run -n baselines python scripts/eval_acc.py`).

    Its output is depth-like rather than disparity-like, so it enters as
    space="depth" -- but it is then aligned by the SAME gauges as every other
    row. scripts/eval_baseline_da3.py fitted its affine in depth space instead,
    which suits this model better and is exactly the per-model protocol A2
    exists to end; the median and scale gauges are reported alongside so a
    reader can see what the choice costs it.
    """
    import numpy as np
    import torch.nn.functional as F
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(name).to(device=dev)

    @torch.no_grad()
    def run(clip):
        T = clip.shape[0]
        imgs = [(f.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                for f in clip]
        out = []
        for i in range(0, T, chunk):
            d = torch.from_numpy(
                np.asarray(model.inference(imgs[i:i + chunk]).depth)).float()
            out.append(F.interpolate(d.unsqueeze(1), size=clip.shape[-2:],
                                     mode="bilinear", align_corners=False))
        return torch.cat(out), {"active": 1.0}

    return run, "depth", model, {}


def hf_runner(name, dev, infer_size=None, chunk=16):
    """A single-frame HF depth model, run frame by frame at its own input
    resolution and resampled back onto the manifest's grid.

    infer_size=None keeps the processor's official resolution (DPT-Large 384,
    DA V2 518) -- the model's quality ceiling. Setting it to the manifest size
    is the like-for-like comparison. The processor's ensure_multiple_of still
    applies (518 -> 252 for a DINOv2 patch grid, not 256), so the effective
    size is printed rather than assumed.
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(name)
    if infer_size is not None:
        proc.size = {"height": infer_size, "width": infer_size}
    net = AutoModelForDepthEstimation.from_pretrained(name).to(dev).eval()
    zoe = type(proc).__name__.startswith("ZoeDepth")

    eff = {}

    @torch.no_grad()
    def run(clip):
        T, _, H, W = clip.shape
        imgs = [Image.fromarray((f.permute(1, 2, 0).numpy() * 255).astype("uint8"))
                for f in clip.cpu()]
        out = []
        for i in range(0, T, chunk):
            batch = imgs[i:i + chunk]
            inp = proc(images=batch, return_tensors="pt").to(dev)
            eff.setdefault("size", tuple(inp["pixel_values"].shape[-2:]))
            pp = {"target_sizes": [(H, W)] * len(batch)}
            if zoe:
                pp["source_sizes"] = [(H, W)] * len(batch)
            post = proc.post_process_depth_estimation(net(**inp), **pp)
            out += [p["predicted_depth"] for p in post]
        return torch.stack(out).unsqueeze(1), {"active": 1.0}

    return run, ("disparity" if not zoe else "depth"), net, eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True,
                    help="ours:<ckpt path> | hf:<huggingface model id> | "
                         "da3:<DA3 checkpoint> (needs the `baselines` env)")
    ap.add_argument("--space", default=None, choices=["depth", "disparity"],
                    help="what the model outputs; default per model family "
                         "(ours/ZoeDepth metric depth, other HF disparity)")
    ap.add_argument("--infer-size", type=int, default=None,
                    help="HF only: force the processor's input resolution. "
                         "Default = the model's official one (A4's ceiling "
                         "row); pass the manifest size for the fair row.")
    ap.add_argument("--align", action="append", choices=list(MODES),
                    help="repeatable; default historical three GT-fit gauges. "
                         "Use --align none for GT-unadjusted metric depth.")
    ap.add_argument("--per-frame", action="store_true",
                    help="fit the gauge on each frame instead of once per "
                         "clip. PLAN_ACC §1.3 asks for both: the clip-wide fit "
                         "carries scale/shift drift, the per-frame one removes "
                         "it and leaves depth-shape error. A long-stream number "
                         "quoted from only one of them is unreadable.")
    ap.add_argument("--temporal", action="store_true",
                    help="also compute OPW/TCE (a RAFT pass per clip, more "
                         "expensive than every other metric combined). G1 is "
                         "an accuracy gate and does not read them.")
    ap.add_argument("--keyframe-every", type=int, default=None)
    ap.add_argument("--tau", type=float, default=None,
                    help="ours: override the detector threshold (0 = every "
                         "patch active). PLAN_ACC A5.")
    ap.add_argument("--reset-every", type=int, default=0,
                    help="ours: restart the streaming state every N frames "
                         "(1 = single-frame mode). PLAN_ACC A5.")
    ap.add_argument("--bypass-temporal", action="store_true",
                    help="ours: make the TemporalBlocks identity. PLAN_ACC A5.")
    ap.add_argument("--sharp", action="store_true",
                    help="also score PLAN.md §3.2's sharpness suite on these "
                         "sealed clips, so P1 and P2 are measured on the same "
                         "clip set instead of on two different protocols. "
                         "Computed after scaleshift alignment; edge AbsRel "
                         "is not alignment-free.")
    ap.add_argument("--regions", action="store_true",
                    help="also break the error down by depth-edge band, "
                         "dynamic/static and near/mid/far (PLAN_ACC A6). "
                         "Costs a RAFT pass per clip.")
    ap.add_argument("--tag", default=None, help="name of the JSON dump")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--final-test", action="store_true",
                    help="evaluate the reserved test only after model selection is frozen")
    ap.add_argument("--reps", type=int, default=10000, help="bootstrap draws")
    args = ap.parse_args()
    role = check_final_test(args.manifest, args.final_test)
    if args.per_frame and args.temporal:
        ap.error("per-frame GT fitting removes temporal scale variation; use clip alignment for temporal metrics")
    if args.sharp and SHARP_GAUGE not in (args.align or LEGACY_MODES):
        ap.error("--sharp requires --align scaleshift (edge AbsRel is gauge-dependent)")
    manifest_meta = json.loads(Path(args.manifest).read_text())
    origin = provenance(args.model, args.manifest)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kind, name = args.model.split(":", 1)
    kw = {"keyframe_every": args.keyframe_every} if args.keyframe_every else {}
    if kind == "ours":
        run, space, net, eff = ours_runner(
            name, dev, tau=args.tau, reset_every=args.reset_every,
            bypass_temporal=args.bypass_temporal, **kw)
    elif kind == "hf":
        run, space, net, eff = hf_runner(name, dev, args.infer_size)
    elif kind == "da3":
        run, space, net, eff = da3_runner(name, dev)
    else:
        sys.exit(f"unknown model kind {kind!r} (want ours:, hf: or da3:)")
    space = args.space or space
    n_par = sum(p.numel() for p in net.parameters()) / 1e6
    modes = list(dict.fromkeys(args.align or LEGACY_MODES))
    if "none" in modes and space != "depth":
        ap.error("--align none requires a metric-depth model; relative disparity has no calibrated metres")
    gname = lambda m: f"{m}/frame" if args.per_frame and m != "none" else m

    sources = load_manifest(args.manifest)
    # "latest.pt" identifies nothing once three arms have one: name the run by
    # its work dir, which is what every table and the sharpness gate key on
    label = (Path(name).parent.name or Path(name).name if kind == "ours"
             else name.split("/")[-1])
    tag = args.tag or (f"{label}-{Path(args.manifest).stem}"
                       + (f"-in{args.infer_size}" if args.infer_size else ""))

    # per_clip[mode][source][metric] -> list. Everything is kept so a different
    # statistic never needs the model re-run.
    per_clip = {m: {s: {} for s, _ in sources} for m in modes}
    fails = {m: {s: {"align": 0, "catastrophic": 0} for s, _ in sources}
             for m in modes}
    sums = {m: {s: [] for s, _ in sources} for m in modes}
    reg = {m: {s: {} for s, _ in sources} for m in modes}
    sharp = {s: {} for s, _ in sources}
    no_gt, active, active_all, t0 = 0, [], [], time.time()
    clip_ids = {s: [] for s, _ in sources}

    for src, ds in sources:
        declared = [c for c in manifest_meta["clips"] if c["source"] == src]
        for i in range(len(ds)):
            clip, gt, valid = ds[i]
            valid = (valid.bool() & torch.isfinite(gt) & (gt > 0)).float()
            if not bool(valid.any()):
                no_gt += 1
                continue
            gt = torch.where(valid.bool(), gt, torch.zeros_like(gt))
            pred, extra = run(clip)
            c = declared[i]
            clip_ids[src].append({"manifest_index_within_source": i,
                                 "sequence": c.get("sequence"),
                                 "first_rgb": c["pairs"][0][0]})
            # everything downstream on one device, and that device is the GPU:
            # the scorer's RAFT pass on CPU cost more than the models did
            clip, pred = clip.to(dev), pred.to(dev)
            gt, valid = gt.to(dev), valid.to(dev)
            active.append(extra["active"])
            active_all.append(extra.get("active_all", extra["active"]))
            rmask = region_masks(clip, gt, valid) if args.regions else None
            for mode in modes:
                a = align(pred, gt, valid, mode, space, args.per_frame)
                if a is None:
                    raise RuntimeError("alignment unexpectedly rejected a clip with valid GT")
                depth, info = a
                sc = clip_scores(clip, depth, gt, valid,
                                 temporal=args.temporal)
                if sc is None:
                    raise RuntimeError("scoring unexpectedly rejected a clip with valid GT")
                fails[mode][src]["align"] += int(info["failed"])
                fails[mode][src]["catastrophic"] += int(sc["absrel"] > 1.0)
                sums[mode][src].append(sc.pop("_pooled"))
                sc["neg_frac"] = info["neg_frac"]
                for k, v in sc.items():
                    per_clip[mode][src].setdefault(k, []).append(v)
                if args.sharp and mode == SHARP_GAUGE:
                    # Score aligned depth in a declared gauge. Edge AbsRel
                    # depends on the fit, and inversion/clamping can also
                    # affect shape metrics; do not claim unconditional invariance.
                    g2 = gt.reshape(-1, 1, *gt.shape[-2:])
                    v2 = valid.reshape(-1, 1, *valid.shape[-2:])
                    for k, x in sharpness_scores(
                            depth.reshape(-1, 1, *depth.shape[-2:]),
                            g2, v2).items():
                        if x == x:                   # NaN: no boundary in clip
                            sharp[src].setdefault(k, []).append(x)
                if rmask is not None:
                    for rname, s in region_scores(depth, gt, rmask).items():
                        t = reg[mode][src].setdefault(
                            rname, {"rel": 0.0, "d1": 0.0, "px": 0.0})
                        for k in t:
                            t[k] += s[k]
            if (i + 1) % 50 == 0:
                print(f"  {src} {i+1}/{len(ds)}", file=sys.stderr)

    empty = [s for s, _ in sources if not clip_ids[s]]
    if empty:
        raise ValueError(f"no scorable GT clips for sources: {empty}; cannot form a balanced table")
    dt = time.time() - t0
    # each field tested against its OWN default: `tau=0` is the dense A5 arm
    # and is exactly the value a truthiness test would drop from the header
    mode5 = ((f" tau={args.tau}" if args.tau is not None else "")
             + (f" reset_every={args.reset_every}" if args.reset_every else "")
             + (" bypass_temporal" if args.bypass_temporal else "")
             + (f" keyframe_every={args.keyframe_every}"
                if args.keyframe_every else ""))
    head = (f"{label} ({n_par:.1f}M) space={space}{mode5} "
            f"infer_size={args.infer_size or ('native' if kind == 'ours' else 'official')}"
            f"{'->' + 'x'.join(map(str, eff['size'])) if eff.get('size') else ''} "
            f"manifest={args.manifest} clips={sum(len(d) for _, d in sources)} "
            f"active={sum(active)/max(len(active),1)*100:.1f}% "
            f"{dt:.0f}s" + (f" (+{no_gt} no-GT)" if no_gt else ""))
    lines = [head, ""]

    for mode in modes:
        lines.append(f"[align={gname(mode)}]")
        lines.append(f"  {'source':<12} {'pooled':>8} {'mean':>8} {'median':>8} "
                     f"{'trim10':>8} {'P90':>8} {'P95':>8} {'max':>9} "
                     f"{'d1':>7} {'fail':>5} {'cat':>4} {'n':>5}")
        for src, _ in sources:
            r = robust(per_clip[mode][src]["absrel"])
            pl = pooled(sums[mode][src])
            f = fails[mode][src]
            lines.append(
                f"  {src:<12} {pl['absrel']:8.4f} {r['mean']:8.4f} "
                f"{r['median']:8.4f} {r['trim10']:8.4f} {r['p90']:8.4f} "
                f"{r['p95']:8.4f} {r['max']:9.4f} {pl['delta1']:7.4f} "
                f"{f['align']:5d} {f['catastrophic']:4d} {r['n']:5d}")
        # MEAN(src): the dataset-balanced number G1 is written against. Equal
        # weight per source, because Bonn has 4.5x TUM's clips and G1 is a
        # statement about both domains rather than about the bigger one.
        bal = {k: sum(robust(per_clip[mode][s]["absrel"])[k]
                      for s, _ in sources) / len(sources)
               for k in ("mean", "median", "trim10", "p90", "p95", "max")}
        bpool = sum(pooled(sums[mode][s])["absrel"]
                    for s, _ in sources) / len(sources)
        bd1 = sum(pooled(sums[mode][s])["delta1"]
                  for s, _ in sources) / len(sources)
        tf = sum(fails[mode][s]["align"] for s, _ in sources)
        tc = sum(fails[mode][s]["catastrophic"] for s, _ in sources)
        n = sum(len(per_clip[mode][s]["absrel"]) for s, _ in sources)
        lines.append(
            f"  {'MEAN(src)':<12} {bpool:8.4f} {bal['mean']:8.4f} "
            f"{bal['median']:8.4f} {bal['trim10']:8.4f} {bal['p90']:8.4f} "
            f"{bal['p95']:8.4f} {bal['max']:9.4f} {bd1:7.4f} "
            f"{tf:5d} {tc:4d} {n:5d}")
        for stat in ("mean", "median"):
            lo, hi = boot_ci({s: per_clip[mode][s]["absrel"] for s, _ in sources},
                             reps=args.reps, stat=stat)
            lines.append(f"  95% CI of balanced clip {stat:<6} AbsRel: "
                         f"[{lo:.4f}, {hi:.4f}]")
        if args.regions:
            lines.append(f"  {'region':<10}" + "".join(
                f"{s:>18}" for s, _ in sources) + f"{'MEAN(src)':>18}")
            lines.append(f"  {'':10}" + f"{'AbsRel  d1   px%':>18}"
                         * (len(sources) + 1))
            for name in REGIONS:
                cells, accum = [], []
                for src, _ in sources:
                    t = reg[mode][src].get(name)
                    tot = reg[mode][src]["all"]["px"] or 1.0
                    if not t or not t["px"]:
                        cells.append(f"{'--':>18}")
                        continue
                    a, d = t["rel"] / t["px"], t["d1"] / t["px"]
                    accum.append((a, d))
                    cells.append(f"{a:8.4f}{d:6.3f}{t['px']/tot*100:4.0f}")
                m = (f"{sum(a for a, _ in accum)/len(accum):8.4f}"
                     f"{sum(d for _, d in accum)/len(accum):6.3f}"
                     f"{'':4}") if accum else f"{'--':>18}"
                lines.append(f"  {name:<10}" + "".join(cells) + m)
        lines.append("")

    # per source, then balanced: one source's clip count must not set the
    # sharpness number any more than it sets AbsRel
    sharp_mean = {s: {k: sum(v) / len(v) for k, v in d.items() if v}
                  for s, d in sharp.items()} if args.sharp else None
    if args.sharp:
        keys = ("absrel_edge", "grad_ratio", "boundary_f1",
                "boundary_precision", "boundary_recall", "flat_tv",
                "overshoot")
        nan = float("nan")
        row = lambda d: "".join(f"{d.get(k, nan):>19.4f}" for k in keys)
        lines.append(f"  sharpness (gauge={gname(SHARP_GAUGE)}; edge AbsRel is alignment-dependent)")
        lines.append(f"  {'source':<10}" + "".join(f"{k:>19}" for k in keys))
        for src, _ in sources:
            lines.append(f"  {src:<10}" + row(sharp_mean[src]))
        lines.append(f"  {'balanced':<10}" + row(
            {k: sum(sharp_mean[s].get(k, nan) for s, _ in sources) / len(sources)
             for k in keys}) + "\n")

    print("\n".join(lines))
    if provenance(args.model, args.manifest) != origin:
        raise RuntimeError("checkpoint, config, manifest or evaluation code changed during this run")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ren = {m: gname(m) for m in modes}
    (out_dir / f"{tag}.json").write_text(json.dumps({
        "protocol_version": VERSION, "provenance": origin, "dataset_role": role,
        "runtime_versions": runtime_versions(),
        "evaluation_options": {k: str(v) if isinstance(v, Path) else v
                               for k, v in vars(args).items()},
        "device": dev, "parameter_dtype": str(next(net.parameters()).dtype),
        "bootstrap_seed": 0,
        "scoring_size": manifest_meta["size"], "temporal_computed": args.temporal,
        "clip_ids": clip_ids,
        "hf_revision": getattr(getattr(net, "config", None), "_commit_hash", None),
        "label": label, "model": args.model, "space": space,
        "infer_size": args.infer_size, "effective_size": eff.get("size"),
        "params_m": n_par,
        "manifest": args.manifest, "n_par": n_par, "mode": mode5.strip(),
        "per_clip": {ren[m]: v for m, v in per_clip.items()},
        "fails": {ren[m]: v for m, v in fails.items()}, "no_gt": no_gt,
        "regions": {ren[m]: v for m, v in reg.items()} if args.regions else None,
        "sharp": sharp_mean,
        # the pixel-pooled per-source numbers the gate reads, so scripts/
        # acc_gate.py never has to re-derive them from the per-clip lists and
        # get a subtly different weighting than the table it is checking
        "pooled": {ren[m]: {s: pooled(sums[m][s]) for s, _ in sources}
                   for m in modes},
        "active": sum(active) / max(len(active), 1),
        "active_all_frames": sum(active_all) / max(len(active_all), 1),
    }))
    with open(out_dir / "table.txt", "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"-> {out_dir / f'{tag}.json'}\n-> {out_dir / 'table.txt'}")


if __name__ == "__main__":
    main()
