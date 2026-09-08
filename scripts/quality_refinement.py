"""Small, development-only, reproducible frozen-v11 refinement screen.

No final test, no promotion, no edits to the original training/evaluation code.
Run from repository root. Outputs are exclusively inside a new --out directory.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.model import from_checkpoint, checkpoint_config
from sokkanaem.data import build_mixed, load_manifest
from sokkanaem.protocol import check_final_test, file_record
from sokkanaem.alignment import align
from sokkanaem.sharpness import sharpness_scores
from sokkanaem.bounded_refiner import BoundedDepthRefiner, refinement_loss


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def paths(ds, i):
    seq, start, stride = ds.clips[i]
    return seq[start:start + ds.T * stride:stride]


def scene(pair):
    return str(Path(pair[0]).parent.parent)


@torch.no_grad()
def extract(model, ds, i, source):
    rgb, gt, valid = ds[i]
    captured = []
    hook = model.decoder.head.register_forward_pre_hook(
        lambda mod, args: captured.append(args[0].detach().cpu().half()))
    state, preds = None, []
    try:
        for frame in rgb:
            pred, state, _ = model.step(frame[None].cuda(), state)
            preds.append(pred.cpu())
    finally:
        hook.remove()
    assert len(captured) == len(rgb)
    return {"rgb": rgb.half(), "gt": gt, "valid": valid.bool(),
            "coarse": torch.cat(preds), "features": torch.cat(captured),
            "source": source, "scene": scene(paths(ds, i)[0]),
            "pairs": paths(ds, i)}


def score(pred, sample):
    g, v = sample["gt"].cuda(), sample["valid"].cuda()
    v = v & (g < 150) & torch.isfinite(g)
    out = {"absrel_raw": float(((pred - g).abs() / g.clamp_min(1e-6))[v].mean())}
    aligned, fit = align(pred, g, v, mode="median", space="depth", per_frame=False)
    out["absrel_clip_median"] = float(((aligned - g).abs() / g.clamp_min(1e-6))[v].mean())
    # The same unaligned positive metric depths go to shape scoring for EVERY candidate.
    out.update({k: float(x) for k, x in sharpness_scores(pred, g, v).items()})
    if not all(np.isfinite(x) for x in out.values()):
        raise ValueError("undefined metric: refusing to silently drop a clip")
    return out


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["scene"])].append(row["metrics"])
    by_source = defaultdict(list)
    for (src, seq), vals in grouped.items():
        by_source[src].append({k: float(np.mean([r[k] for r in vals])) for k in vals[0]})
    means = {src: {k: float(np.mean([r[k] for r in vals])) for k in vals[0]}
             for src, vals in by_source.items()}
    balanced = {k: float(np.mean([r[k] for r in means.values()])) for k in next(iter(means.values()))}
    return {"balanced": balanced, "sources": means, "clips": rows}


@torch.no_grad()
def evaluate(samples, refiner=None):
    rows = []
    for s in samples:
        p = s["coarse"].cuda()
        if refiner is not None:
            p, _ = refiner(s["rgb"].cuda().float(), p, s["features"].cuda().float())
        rows.append({"source": s["source"], "scene": s["scene"], "metrics": score(p, s)})
    return aggregate(rows)


def gate(base, candidate):
    a, b = base["balanced"], candidate["balanced"]
    checks = {
        "raw_accuracy_nonregression": b["absrel_raw"] <= a["absrel_raw"],
        "median_accuracy_nonregression": b["absrel_clip_median"] <= a["absrel_clip_median"],
        "boundary_f1_plus_0.005": b["boundary_f1"] >= a["boundary_f1"] + .005,
        "overshoot_nonregression": b["overshoot"] <= a["overshoot"],
        "flat_tv_nonregression": b["flat_tv"] <= a["flat_tv"],
        "gradient_upper_bound": b["grad_ratio"] <= 1.0,
        "per_source_raw_regression_at_most_1pct": all(
            candidate["sources"][s]["absrel_raw"] <= x["absrel_raw"] * 1.01
            for s, x in base["sources"].items()),
    }
    return {"checks": checks, "pass": all(checks.values()),
            "note": "Screening gate only; not significance, long-stream or device validation."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", default="work_dirs/v11-longclip-spread-s0/latest.pt")
    ap.add_argument("--manifest", default="manifests/acc_real_L8.json")
    ap.add_argument("--train-per-source", type=int, default=16)
    ap.add_argument("--dev-per-scene", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a fresh output directory; experiments are immutable.")
    assert check_final_test(args.manifest) == "development_validation"
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(20260908); random.seed(20260908); np.random.seed(20260908)
    torch.backends.cudnn.benchmark = False
    cfg = checkpoint_config(args.checkpoint)
    # Equal source counts, sensor + synthetic. No new teacher or backbone updates.
    specs = [s for s in cfg["data"] if s.split(":")[0] in ("tum", "bonn", "vkitti2")]
    train, _ = build_mixed(specs, holdout=cfg["holdout"], clip_len=4,
                           clip_stride=16, size=256, strict=True, augment=False)
    dev_sets = load_manifest(args.manifest)
    dev_sel = []
    for src, ds in dev_sets:
        groups = defaultdict(list)
        for i in range(len(ds)):
            groups[scene(paths(ds, i)[0])].append(i)
        for seq, indices in groups.items():
            for j in np.linspace(0, len(indices)-1, min(args.dev_per_scene, len(indices))).astype(int):
                dev_sel.append((src, ds, indices[j]))
    train_sel = []
    rng = np.random.default_rng(20260908)
    for spec, ds in zip(specs, train.datasets):
        for i in rng.choice(len(ds), size=min(args.train_per_source, len(ds)), replace=False):
            train_sel.append((spec.split(":")[0], ds, int(i)))
    dev_frames = {str(Path(p).resolve()) for _, ds, i in dev_sel for pair in paths(ds, i) for p in pair}
    train_frames = {str(Path(p).resolve()) for _, ds, i in train_sel for pair in paths(ds, i) for p in pair}
    assert not dev_frames & train_frames, "train/development overlap"
    selection = {"train": [{"source": s, "pairs": paths(d, i)} for s, d, i in train_sel],
                 "development": [{"source": s, "pairs": paths(d, i)} for s, d, i in dev_sel]}
    write(args.out / "selection.json", selection)
    protocol = {"role": "development_screen_not_final", "seed": 20260908,
                "steps": args.steps, "batch": 8, "lr": .001,
                "variants": {"metric_only": 0.0, "metric_gradient": .5},
                "max_log_delta": .15, "inference": "native 256, K30, reset at clip start",
                "train_clip_len": 4, "development_clip_len": 8,
                "aggregation": "clip -> scene -> source -> equal-source mean",
                "selection": file_record(args.out / "selection.json"),
                "checkpoint": file_record(args.checkpoint),
                "config": file_record(Path(args.checkpoint).with_name("config.toml")),
                "manifest": file_record(args.manifest),
                "code": [file_record(p) for p in [__file__, "sokkanaem/bounded_refiner.py",
                          "sokkanaem/model.py", "sokkanaem/data.py", "sokkanaem/sharpness.py",
                          "sokkanaem/alignment.py"]]}
    write(args.out / "protocol.json", protocol)
    model = from_checkpoint(args.checkpoint, "cuda").eval().requires_grad_(False)
    cached = []
    for label, selected in (("train", train_sel), ("development", dev_sel)):
        samples = []
        for j, (src, ds, i) in enumerate(selected):
            samples.append(extract(model, ds, i, src))
            print(f"cache {label} {j+1}/{len(selected)}", flush=True)
        torch.save(samples, args.out / f"{label}_cache.pt")
        cached.append(samples)
    del model
    train_cache, dev_cache = cached
    baseline = evaluate(dev_cache)
    write(args.out / "baseline.json", baseline)
    print("baseline", baseline["balanced"], flush=True)
    # Fair cached feature/raw depth precision: coarse float32, RGB/features float16.
    tensors = {k: torch.cat([s[k] for s in train_cache]).cuda().float()
               for k in ("rgb", "coarse", "features", "gt", "valid")}
    results = {"baseline": baseline, "variants": {}}
    for name, edge_weight in protocol["variants"].items():
        torch.manual_seed(20260908)
        refiner = BoundedDepthRefiner().cuda()
        identity = evaluate(dev_cache, refiner)
        assert identity["balanced"] == baseline["balanced"]
        optimizer = torch.optim.AdamW(refiner.parameters(), lr=.001, weight_decay=.0001)
        history = []
        start = time.monotonic()
        for step in range(1, args.steps + 1):
            idx = torch.randint(len(tensors["rgb"]), (8,), device="cuda")
            x = {k: v[idx] for k, v in tensors.items()}
            pred, delta = refiner(x["rgb"], x["coarse"], x["features"])
            loss, parts = refinement_loss(pred, x["coarse"], x["gt"], x["valid"], delta, edge_weight)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % 100 == 0:
                row = {"step": step, "loss": float(loss.detach()), **parts}
                history.append(row); print(name, row, flush=True)
        refiner.eval()
        measured = evaluate(dev_cache, refiner)
        outcome = {"metrics": measured, "gate": gate(baseline, measured),
                   "seconds": time.monotonic()-start, "history": history,
                   "parameters": sum(p.numel() for p in refiner.parameters())}
        torch.save({"refiner": refiner.state_dict(), "max_log_delta": .15,
                    "base_checkpoint_sha256": protocol["checkpoint"]["sha256"],
                    "variant": name, "steps": args.steps}, args.out / f"{name}.pt")
        results["variants"][name] = outcome
        write(args.out / f"{name}.json", outcome)
        print(name, measured["balanced"], outcome["gate"], flush=True)
    results["promoted"] = False
    results["next"] = "Independent expanded development, long-stream and latency validation required even if screen passes."
    assert file_record(args.checkpoint) == protocol["checkpoint"]
    write(args.out / "results.json", results)


if __name__ == "__main__":
    main()
