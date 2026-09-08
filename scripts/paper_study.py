"""Execute paper tasks 5–8 on DEVELOPMENT validation, never the reserved test.

Frozen source files and weights are read-only. New code and pinned baselines
are recorded in each run's study.json. One complete clip is one resumable row.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokkanaem import from_checkpoint
from sokkanaem.data import load_manifest
from sokkanaem.metrics import _flow
from sokkanaem.protocol import (EVALUATION_FILES, check_final_test, file_record,
                                runtime_versions)
from sokkanaem.study import (BENCH_ARMS, HFDepth, LONG_ARMS, grid_from_flow,
    native_predict, original_views, regions_from_flow, score_prediction)

BASE = Path("work_dirs/v11-longclip-spread-s0/latest.pt")
HF = {
    "dpt": ("Intel/dpt-large", "bc15f29aa3a80d532f2ed650b5e16ac48d8958f9"),
    "da2": ("depth-anything/Depth-Anything-V2-Small-hf", "5426e4f0f36572d16453bbda7a8389317b1bef99"),
    "zoe": ("Intel/zoedepth-nyu-kitti", "f364d4c7936e91f465abba182208dd68142bf0ca"),
}


def snapshot_path(name, revision):
    return Path("/home/hyunsu/.cache/huggingface/hub") / ("models--" + name.replace("/", "--")) / "snapshots" / revision


def dump(path, obj):
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"refusing to overwrite an existing record: {path}")
    else:
        path.write_text(text)


def audit(out, smoke):
    freeze = json.loads(Path("paper/freeze/baseline.json").read_text())
    for rec in [freeze["checkpoint"], freeze["config"], *freeze["code"]]:
        if file_record(rec["path"])["sha256"] != rec["sha256"]:
            raise RuntimeError(f"frozen v1 artifact changed: {rec['path']}")
    files = [*EVALUATION_FILES, "sokkanaem/study.py", "scripts/paper_study.py",
             "paper/WORK_PLAN.md", "paper/freeze/baseline.json"]
    baselines = {}
    for tag, (name, rev) in HF.items():
        snapshot = snapshot_path(name, rev)
        baselines[tag] = {"id": name, "revision": rev, "snapshot": str(snapshot),
                          "future_frames": False,
                          "files": [file_record(p) for p in sorted(snapshot.iterdir()) if p.is_file()]}
    manifests = [Path(f"manifests/acc_real_L{length}.json") for length in (8, 256)]
    for p in manifests:
        if check_final_test(p) != "development_validation":
            raise ValueError("this study must not open the final test")
    data_paths = sorted({p for manifest in manifests for c in json.loads(manifest.read_text())["clips"]
                         for pair in c["pairs"] for p in pair})
    data_record = {"files": [file_record(p) for p in data_paths]}
    dump(out / "data_files.json", data_record)
    contract = {"kind": "development_diagnostic", "smoke": smoke,
                "base": freeze["checkpoint"], "sources": [file_record(p) for p in files],
                "manifests": [file_record(p) for p in manifests],
                "data_inventory": file_record(out / "data_files.json"),
                "baselines": baselines, "long_arms": LONG_ARMS,
                "common_input": "frozen 256 RGB -> processor requested 256; record actual grid",
                "official_input": "original-resolution scoring ROI -> official processor; scoring at 256",
                "flow": file_record("/home/hyunsu/.cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth"),
                "precision": "FP32, eager, TF32 off", "environment": runtime_versions(),
                "device": torch.cuda.get_device_name(0), "training_performed": False,
                "final_test_used": False, "bench": {"frames": 64, "warmups": 1, "repeats": 3,
                    "selection": "earliest L256 clip of each development sequence, same masks and frames"}}
    dump(out / "study.json", contract)
    return contract


def records(length, smoke=False):
    path = Path(f"manifests/acc_real_L{length}.json")
    doc = json.loads(path.read_text())
    for source, dataset in load_manifest(path):
        declared = [c for c in doc["clips"] if c["source"] == source]
        for i, clip in enumerate(declared):
            if smoke and i >= 1:
                break
            identity = {"id": f"{source}:{i}", "source": source, "index": i,
                        "sequence": Path(clip["pairs"][0][0]).parent.parent.name,
                        "first_rgb": clip["pairs"][0][0], "pairs": clip["pairs"]}
            yield identity, dataset, i


def completed(path):
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate completed clips: {path}")
    return set(ids)


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def models_for_long():
    models = {}
    for name, arm in LONG_ARMS.items():
        kwargs = {k: v for k, v in arm.items() if k in
                  ("keyframe_every", "spatial_cache", "temporal_cache", "gate_mode")}
        models[name] = from_checkpoint(BASE, "cuda", **kwargs).eval()
    return models


@torch.no_grad()
def evaluate(out, phase, smoke=False):
    length = 8 if phase == "accuracy" else 256
    path = out / f"{phase}.jsonl"
    done = completed(path)
    todo = [(c, ds, i) for c, ds, i in records(length, smoke) if c["id"] not in done]
    if not todo:
        print(f"{phase}: all clips already complete", flush=True)
        return
    if phase == "accuracy":
        models = {"sparse_k30": from_checkpoint(BASE, "cuda").eval()}
        baselines = {tag: HFDepth(str(snapshot_path(*spec))) for tag, spec in HF.items()}
        import inspect
        dump(out / "baseline_loading.json", {tag: {
            "loading": model.loading,
            "guard": "missing-parameter modules raise if executed",
            "model_code": file_record(inspect.getfile(type(model.model))),
            "processor_code": file_record(inspect.getfile(type(model.processor))),
            "official_processor": model.processor.to_dict(),
        } for tag, model in baselines.items()})
    else:
        models, baselines = models_for_long(), {}
    start = time.monotonic()
    for ordinal, (identity, ds, index) in enumerate(todo):
        frames, gt, valid = ds[index]
        valid = (valid.bool() & torch.isfinite(gt) & (gt > 0)).float()
        row = {**identity, "clip_len": length}
        if not bool(valid.any()):
            row.update(no_gt=True, models={})
            append(path, row)
            continue
        gt = torch.where(valid.bool(), gt, torch.zeros_like(gt))
        frames_gpu, gt, valid = frames.cuda(), gt.cuda(), valid.cuda()
        flow = _flow(frames_gpu[1:], frames_gpu[:-1], chunk=16)
        grid, inb = grid_from_flow(flow)
        regions = regions_from_flow(gt, valid, flow) if phase == "accuracy" else None
        row.update(no_gt=False, models={})
        forced, baseline_prediction = None, None
        for name, model in models.items():
            arm = LONG_ARMS[name]
            pred, masks, mac = native_predict(model, frames_gpu, arm, forced, count=True)
            if name == "sparse_k30":
                forced = masks
                baseline_prediction = pred
                row["shared_mask"] = masks.to(torch.uint8).cpu().tolist()
            if arm.get("forced") and not torch.equal(masks, forced):
                raise AssertionError(f"mask mismatch in {name}")
            if name == "both_cache_replay" and not torch.allclose(pred, baseline_prediction, atol=1e-5, rtol=1e-5):
                raise AssertionError("fixed-mask replay changed the baseline prediction")
            row["models"][name] = {"space": "depth", "params": sum(p.numel() for p in model.parameters()),
                "future_frames": False, "effective_size": [256, 256],
                "active_all": masks.mean().item(), "active_post_first": masks[1:].mean().item(),
                "linear_conv_gmac_per_frame": mac / length / 1e9,
                "gauges": score_prediction(frames_gpu, pred, gt, valid, "depth", grid, inb,
                                            regions, long=phase == "long", metric=True)}
        if baselines:
            originals = original_views(identity["pairs"])
            for tag, model in baselines.items():
                for official in (False, True):
                    regime = "official" if official else "common"
                    pred = model.predict(frames, originals, official)
                    row["models"][f"{tag}_{regime}"] = {
                        "space": model.space, "params": sum(p.numel() for p in model.model.parameters()),
                        "future_frames": False, "effective_size": model.effective[regime],
                        "gauges": score_prediction(frames_gpu, pred, gt, valid, model.space,
                                                    grid, inb, regions, metric=model.metric)}
        append(path, row)
        elapsed = time.monotonic() - start
        print(f"{phase} {ordinal + 1}/{len(todo)} {identity['id']} {identity['sequence']} "
              f"elapsed={elapsed:.0f}s mean={elapsed / (ordinal + 1):.2f}s/clip", flush=True)
    print(f"{phase} complete: {path}", flush=True)


@torch.no_grad()
def benchmark(out, smoke=False):
    path = out / "ablation_latency.json"
    if path.exists():
        print("benchmark already complete", flush=True)
        return
    models = models_for_long()
    selected, seen = [], set()
    for identity, ds, i in records(256, smoke):
        if identity["sequence"] not in seen:
            seen.add(identity["sequence"])
            frames = ds[i][0][:64].cuda()
            _, mask, _ = native_predict(models["sparse_k30"], frames, LONG_ARMS["sparse_k30"])
            selected.append((identity, frames, mask))
    rows = []
    for identity, frames, mask in selected:
        for name in BENCH_ARMS:
            model, arm = models[name], LONG_ARMS[name]
            native_predict(model, frames, arm, mask)  # untimed warmup, clip state reset
            torch.cuda.synchronize()
            timings, peak = [], []
            for _ in range(3):
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                native_predict(model, frames, arm, mask)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000 / len(frames))
                peak.append(torch.cuda.max_memory_allocated() / 2**20)
            rows.append({"sequence": identity["sequence"], "clip_id": identity["id"],
                         "arm": name, "frames": len(frames), "mask_active_all": mask.mean().item(),
                         "window_ms_per_frame": timings,
                         "process_peak_mib": peak})
            print(f"bench {identity['sequence']} {name}: {np.mean(timings):.3f} ms/frame", flush=True)
    dump(path, {"rows": rows, "scope": "GPU-resident FP32 eager fixed-mask predictor; no detector/IO/scoring",
                "statistic": "three 64-frame window means per sequence; not per-frame P95",
                "memory_scope": "whole study process with all native variants resident; not isolated model memory"})


def verify_contract(contract):
    records_to_check = [contract["base"], *contract["sources"], *contract["manifests"],
                        contract["flow"], contract["data_inventory"]]
    for baseline in contract["baselines"].values():
        records_to_check.extend(baseline["files"])
    records_to_check.extend(json.loads(Path(contract["data_inventory"]["path"]).read_text())["files"])
    for item in records_to_check:
        if file_record(item["path"])["sha256"] != item["sha256"]:
            raise RuntimeError(f"study input changed: {item['path']}")
    return len(records_to_check)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("work_dirs/paper_study_5_8"))
    ap.add_argument("--phase", choices=["accuracy", "long", "bench", "all"], default="all")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this actual-data study requires CUDA")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    contract = audit(args.out, args.smoke)
    phases = ("accuracy", "long", "bench") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "bench":
            benchmark(args.out, args.smoke)
        else:
            evaluate(args.out, phase, args.smoke)
        torch.cuda.empty_cache()
        n = verify_contract(contract)
        dump(args.out / f"{phase}_complete.json", {"phase": phase, "verified_file_records": n,
             "smoke": args.smoke, "result": file_record(args.out / ("ablation_latency.json" if phase == "bench" else f"{phase}.jsonl"))})
        print(f"{phase}: verified {n} study inputs after execution", flush=True)


if __name__ == "__main__":
    main()
