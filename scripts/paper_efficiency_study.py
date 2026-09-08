"""Run tasks 9–11 on development data; frozen 1–8 artifacts remain read-only."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paper_study import append, completed, dump, records, verify_contract
from sokkanaem import metrics
from sokkanaem.performance_study import POINTS, DEPS, adapter, run_clip, StageTimer
from sokkanaem.protocol import file_record, runtime_versions
from sokkanaem.study import grid_from_flow, score_prediction

PRIOR = Path("work_dirs/paper_study_5_8")
SEEDS = [Path(f"work_dirs/v11-longclip-spread-s{i}/latest.pt") for i in range(3)]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,memory.used,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def assert_gpu_exclusive():
    import os
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
    others = [int(s.strip()) for s in out.stdout.splitlines() if s.strip().isdigit() and int(s) != os.getpid()]
    if others:
        raise RuntimeError(f"other GPU compute processes are active: {others}")


def dependency_records():
    result = [file_record(DEPS / "midas_v21_small_256.pt")]
    repositories = {}
    for name in ("MiDaS-v2_1", "gen-efficientnet-pytorch"):
        directory = DEPS / name
        rev = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        paths = subprocess.run(["git", "-C", str(directory), "ls-files"],
                               capture_output=True, text=True, check=True).stdout.splitlines()
        repositories[name] = rev
        result.extend(file_record(directory / p) for p in paths)
    return result, repositories


def seed_provenance():
    result = []
    relevant = ("scripts/train.py", "sokkanaem/model.py", "sokkanaem/ssm.py", "sokkanaem/data.py",
                "sokkanaem/losses.py", "sokkanaem/metrics.py", "sokkanaem/ema.py", "sokkanaem/distill.py")
    for path in SEEDS:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        meta = checkpoint["meta"]
        history = {}
        for source in relevant:
            proc = subprocess.run(["git", "show", f"{meta['git_commit']}:{source}"], capture_output=True)
            history[source] = hashlib.sha256(proc.stdout).hexdigest() if proc.returncode == 0 else None
        result.append({"checkpoint": file_record(path), "config": file_record(path.with_name("config.toml")),
                       "step": checkpoint["step"], "meta": meta, "historical_tracked_sources": history})
    reference = {k: v for k, v in result[0]["meta"]["args"].items() if k not in ("seed", "work_dir")}
    if any({k: v for k, v in r["meta"]["args"].items() if k not in ("seed", "work_dir")} != reference
           or r["meta"]["model"] != result[0]["meta"]["model"] for r in result):
        raise ValueError("seed training configurations differ beyond seed/output path")
    return {"runs": result, "parent": file_record(reference["resume"]),
            "scope": "three last-stage 8k fine-tuning seeds from the SAME v10 parent; not full independent training",
            "historical_worktree_caveat": "recorded commit does not attest historical uncommitted changes"}


def audit(out, smoke):
    prior = json.loads((PRIOR / "study.json").read_text())
    verify_contract(prior)
    for phase in ("accuracy", "long", "bench"):
        rec = json.loads((PRIOR / f"{phase}_complete.json").read_text())["result"]
        if file_record(rec["path"])["sha256"] != rec["sha256"]:
            raise ValueError("prior result has changed")
    dependencies, revisions = dependency_records()
    seed = seed_provenance()
    dump(out / "seed_provenance.json", seed)
    files = ["scripts/paper_efficiency_study.py", "sokkanaem/performance_study.py",
             "paper/STUDY_9_11_PLAN.md", str(PRIOR / "study.json"),
             str(PRIOR / "accuracy.jsonl"), str(PRIOR / "long.jsonl"), str(out / "seed_provenance.json")]
    contract = {"role": "development_validation", "smoke": smoke,
                "training_performed": False, "final_test_used": False,
                "points": POINTS, "sources": [file_record(p) for p in files],
                "dependencies": dependencies, "dependency_revisions": revisions,
                "seed_inputs": [v for r in seed["runs"] for v in (r["checkpoint"], r["config"])],
                "parent": seed["parent"], "environment": runtime_versions(),
                "timing": {"batch": 1, "precision": "FP32 eager TF32 off", "torch_threads": 2,
                    "opencv_threads": 1, "warmup_clips": 1, "repeats": 5,
                    "frames": 256, "selection": "first complete L256 per development sequence",
                    "scope": "warm-cache PNG file to CPU depth; model load/GT/flow/scoring/output saving excluded"}}
    dump(out / "study.json", contract)
    return contract


def verify(contract):
    verify_contract(json.loads((PRIOR / "study.json").read_text()))
    entries = [*contract["sources"], *contract["dependencies"], *contract["seed_inputs"], contract["parent"]]
    for record in entries:
        if file_record(record["path"])["sha256"] != record["sha256"]:
            raise ValueError(f"study input changed: {record['path']}")
    return len(entries)


def selected(smoke=False):
    seen, chosen = set(), []
    for identity, dataset, index in records(256):
        if identity["sequence"] not in seen:
            seen.add(identity["sequence"])
            chosen.append((identity, dataset, index))
    return chosen[:1] if smoke else chosen


def clean_gpu():
    metrics._raft = None
    gc.collect()
    torch.cuda.synchronize()
    # CUDA BLAS retains ~8 MiB per thread even with no live model tensors.
    # Clear that runtime workspace as well before the isolation assertion.
    clear_blas = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear_blas is not None:
        clear_blas()
    torch.cuda.empty_cache()


@torch.no_grad()
def score_setup(dataset, index, cap=None):
    frames, gt, valid = dataset[index]
    if cap:
        frames, gt, valid = frames[:cap], gt[:cap], valid[:cap]
    valid = (valid.bool() & torch.isfinite(gt) & (gt > 0)).float()
    if not bool(valid.any()):
        return None
    gt = torch.where(valid.bool(), gt, torch.zeros_like(gt))
    frames, gt, valid = frames.cuda(), gt.cuda(), valid.cuda()
    flow = metrics._flow(frames[1:], frames[:-1], chunk=16)
    grid, inb = grid_from_flow(flow)
    return frames, gt, valid, grid, inb


def result(model, prediction, setup):
    frames, gt, valid, grid, inb = setup
    return {"params": model.params, "space": model.space, "future_frames": False,
            "effective_size": prediction["input_size"], "original_resolution_roi": model.original,
            "prediction_sha256": prediction["prediction_sha256"],
            "active_all": np.mean([t["active"] for t in prediction["telemetry"]]) if model.metric else None,
            "gmc_calls": prediction["telemetry"][-1]["gmc_calls"],
            "gmc_fallbacks": prediction["telemetry"][-1]["gmc_fallbacks"],
            "gauges": score_prediction(frames, prediction["prediction"].cuda(), gt, valid, model.space,
                                        grid, inb, metric=model.metric)}


def quality(out, smoke):
    path = out / "quality.jsonl"
    done = completed(path)
    prior = {r["id"]: r for r in rows(PRIOR / "long.jsonl")}
    points = list(POINTS)
    models = {name: adapter(POINTS[name]) for name in points}
    todo = list(records(256))[:1] if smoke else list(records(256))
    for identity, dataset, index in todo:
        if identity["id"] in done:
            continue
        pairs = identity["pairs"][:8] if smoke else identity["pairs"]
        setup = score_setup(dataset, index, 8 if smoke else None)
        row = {**identity, "clip_len": len(pairs), "no_gt": setup is None, "models": {}}
        if setup is not None:
            for name, model in models.items():
                prediction = run_clip(model, pairs)
                item = result(model, prediction, setup)
                if not smoke and name in prior[identity["id"]]["models"]:
                    old = prior[identity["id"]]["models"][name]
                    for gauge in ("none", "median", "scaleshift"):
                        for metric in ("absrel", "rmse", "delta1", "tce"):
                            a, b = item["gauges"][gauge]["scores"][metric], old["gauges"][gauge]["scores"][metric]
                            if not np.isclose(a, b, rtol=2e-5, atol=1e-6):
                                raise ValueError(f"deployment/old quality mismatch: {identity['id']} {name} {gauge} {metric}: {a} != {b}")
                    item["prior_score_equivalence"] = True
                row["models"][name] = item
                del prediction
                print(f"quality {identity['id']} {name}", flush=True)
        append(path, row)
        del setup
    del models
    clean_gpu()


def seed_quality(out, smoke):
    models = {f"seed{i}": adapter(POINTS["sparse_k30"], checkpoint=SEEDS[i]) for i in (1, 2)}
    for length in (8, 256):
        path = out / f"seeds_L{length}.jsonl"
        done = completed(path)
        todo = list(records(length))[:1] if smoke else list(records(length))
        for identity, dataset, index in todo:
            if identity["id"] in done:
                continue
            setup = score_setup(dataset, index, 8 if smoke else None)
            row = {**identity, "clip_len": 8 if smoke else length, "no_gt": setup is None, "models": {}}
            if setup is not None:
                for name, model in models.items():
                    prediction = run_clip(model, identity["pairs"][:8] if smoke else identity["pairs"])
                    row["models"][name] = result(model, prediction, setup)
                    del prediction
            append(path, row)
            del setup
            if index % 20 == 0 or length == 256:
                print(f"seeds L{length} {identity['id']}", flush=True)
    del models
    clean_gpu()


def latency(out, smoke, profile=False):
    path = out / ("profile.jsonl" if profile else "latency.jsonl")
    done = completed(path)
    reference = {r["id"]: r for r in rows(out / "quality.jsonl")}
    names = list(POINTS)
    np.random.default_rng(20260906).shuffle(names)
    for name in names:
        needed = [(identity, dataset, index) for identity, dataset, index in selected(smoke)
                  if f"{name}:{identity['id']}" not in done]
        if not needed:
            continue
        assert_gpu_exclusive()
        clean_gpu()
        allocation_before_model = torch.cuda.memory_allocated()
        if allocation_before_model > 1024 * 1024:
            raise RuntimeError(f"unexpected GPU tensors before isolated model load: {allocation_before_model}")
        timer = StageTimer() if profile else None
        model = adapter(POINTS[name], timer=timer)
        gpu_before = gpu_state()
        for identity, _, _ in needed:
            pairs = identity["pairs"][:8] if smoke else identity["pairs"]
            run_clip(model, pairs, collect=False)  # complete warmup, state reset again below
            repeats = 1 if profile or smoke else 5
            runs = []
            for repeat in range(repeats):
                assert_gpu_exclusive()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                prediction = run_clip(model, pairs, timing=True, profile=profile, collect=False)
                if prediction["prediction_sha256"] != reference[identity["id"]]["models"][name]["prediction_sha256"]:
                    raise ValueError(f"timed/quality prediction mismatch: {name} {identity['id']} repeat {repeat}")
                prediction.pop("prediction")
                runs.append({**prediction, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
            row = {"id": f"{name}:{identity['id']}", "model": name, "clip_id": identity["id"],
                   "source": identity["source"], "sequence": identity["sequence"], "frames": len(pairs),
                   "runs": runs, "allocation_before_model": allocation_before_model,
                   "params": model.params, "gpu_before": gpu_before, "gpu_after": gpu_state(),
                   "prediction_verified": True, "profile": profile}
            append(path, row)
            print(f"{'profile' if profile else 'latency'} {name} {identity['id']}: "
                  f"{np.mean([np.mean(r['ms']) for r in runs]):.3f} ms/frame", flush=True)
        del model, timer
        clean_gpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("work_dirs/paper_study_9_11"))
    ap.add_argument("--phase", choices=("quality", "latency", "profile", "seeds"), required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this study requires the actual CUDA deployment device")
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    assert_gpu_exclusive()
    contract = audit(args.out, args.smoke)
    start = time.monotonic()
    {"quality": quality, "seeds": seed_quality,
     "latency": latency, "profile": lambda out, smoke: latency(out, smoke, profile=True)}[args.phase](args.out, args.smoke)
    n = verify(contract)
    paths = [args.out / f"seeds_L{length}.jsonl" for length in (8, 256)] if args.phase == "seeds" else [args.out / f"{args.phase}.jsonl"]
    dump(args.out / f"{args.phase}_complete.json", {"phase": args.phase, "smoke": args.smoke,
        "verified_additional_records": n, "results": [file_record(p) for p in paths]})
    print(f"{args.phase} complete, inputs verified; elapsed {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
