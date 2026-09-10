"""Audit K2 screen: cache control, refresh phase, repeated latency, longer dev.

Same development scenes, NOT an independent scene/generalization or final test.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact, MotionStream
from scripts.qstream_study import evaluate, quality_gate
from sokkanaem.data import load_manifest
from sokkanaem.protocol import file_record, check_final_test
from sokkanaem.qdelta_ssm import QDeltaSSM
from sokkanaem.qstream import ChangeAwareQ0
from sokkanaem.qrgb_refiner import QRGBRefiner, RefinedStream


class ShiftedPhase(MotionStream):
    @torch.no_grad()
    def step(self, frame, state=None):
        n = 0 if state is None else state["n"]
        if n == 1:
            state = {**state, "age": self.refresh_every-1}
        pred, state, info = super().step(frame, state)
        state["n"] = n+1
        return pred, state, info


@torch.no_grad()
def latency_audit(models, samples):
    frames = [s["rgb"].cuda() for s in samples]
    result = {k: [] for k in models}
    keys = list(models)
    for repeat in range(3):
        for key in keys[repeat:]+keys[:repeat]:
            model = models[key]; state = None
            for t in range(64): _, state, _ = model.step(frames[0][t % len(frames[0]):t % len(frames[0])+1], state)
            values = []; refresh = []; update = []
            for rgb in frames:
                state = None
                for frame in rgb:
                    torch.cuda.synchronize(); start = time.perf_counter()
                    _, state, info = model.step(frame[None], state)
                    torch.cuda.synchronize(); elapsed = (time.perf_counter()-start)*1000
                    values.append(elapsed)
                    (refresh if info["full_refresh"] else update).append(elapsed)
            result[key].append({"repeat": repeat, "mean_ms": statistics.mean(values),
                                "p95_ms": float(np.quantile(values, .95)),
                                "refresh_mean_ms": statistics.mean(refresh),
                                "update_mean_ms": statistics.mean(update) if update else None,
                                "per_frame_ms": values})
            print("latency", key, repeat, result[key][-1]["mean_ms"], flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    parent = Path("work_dirs/qquality_feature_20260908")
    pp = json.loads((parent/"protocol.json").read_text())
    torch.set_num_threads(4); torch.manual_seed(736); torch.backends.cudnn.benchmark = False
    exact = load_exact(pp["checkpoint"])
    adapter = QDeltaSSM().cuda().eval()
    adapter.load_state_dict(torch.load(parent/"adapter.pt", map_location="cuda", weights_only=False)["adapter"])
    refiner_path = Path("work_dirs/qquality_rgb_20260908/refiner.pt")
    refiner = QRGBRefiner().cuda().eval()
    refiner.load_state_dict(torch.load(refiner_path, map_location="cuda", weights_only=False)["refiner"])
    dev = torch.load(pp["development"]["path"], weights_only=False)
    manifest = Path("manifests/acc_real_L256.json")
    assert check_final_test(manifest) == "development_validation"
    seen = {str(Path(p).resolve()) for s in dev for pair in s["pairs"] for p in pair}
    selected = []
    for src, ds in load_manifest(manifest):
        groups = defaultdict(list)
        for i in range(len(ds)): groups[q.scene(q.paths(ds, i)[0])].append(i)
        for scene, indices in groups.items():
            # Lowest L32 frame overlap; tie closest to group midpoint, then index.
            def rank(i):
                paths = {str(Path(p).resolve()) for pair in q.paths(ds, i) for p in pair}
                return len(paths & seen), abs(i-indices[len(indices)//2]), i
            i = min(indices, key=rank)
            selected.append((src, ds, i, rank(i)[0]))
    assert len(selected) == 4
    train = torch.load("work_dirs/qdelta_20260908/teacher_cache.pt", weights_only=False)
    train_paths = {str(Path(p).resolve()) for s in train for pair in s["pairs"] for p in pair}
    long_paths = {str(Path(p).resolve()) for _, ds, i, _ in selected for pair in q.paths(ds, i) for p in pair}
    assert not train_paths & long_paths
    args.out.mkdir(parents=True)
    protocol = {"role": "post_screen_diagnostic_same_development_scenes_NOT_independent_test",
                "checkpoint": pp["checkpoint"], "adapter": file_record(parent/"adapter.pt"),
                "manifest": file_record(manifest), "development": pp["development"],
                "long_selection": [{"source": s, "pairs": q.paths(d, i), "L32_overlap_paths": n} for s, d, i, n in selected],
                "policies": ["dense", "featureK2", "featureK2_shifted", "holdK2", "rgbK2"],
                "refiner": file_record(refiner_path),
                "temporal_caveat": "K2 refresh resets h; only one SSM step per refresh, hence no carried-state contribution is possible",
                "timing": "3 rotated-order repeats;64 warmup frames per policy;256 timed GPU-resident frames;FP32;IO/H2D excluded",
                "code": [file_record(p) for p in [__file__, "scripts/qquality_study.py", "scripts/qstream_study.py",
                          "sokkanaem/qstream.py", "sokkanaem/qdelta_ssm.py", "sokkanaem/qquality.py",
                          "scripts/quality_refinement.py", "sokkanaem/sharpness.py", "sokkanaem/qrgb_refiner.py"]]}
    q.write(args.out/"protocol.json", protocol)
    models = {"dense": exact, "featureK2": MotionStream(exact, adapter, refresh_every=2),
              "featureK2_shifted": ShiftedPhase(exact, adapter, refresh_every=2),
              "holdK2": ChangeAwareQ0(exact, refresh_every=2, periodic_only=True),
              "rgbK2": RefinedStream(MotionStream(exact, adapter, refresh_every=2), refiner)}
    timing = latency_audit({k: models[k] for k in ("dense", "featureK2", "holdK2", "rgbK2")}, dev)
    q.write(args.out/"latency.json", timing)
    result = {"L32": {}, "L256": {}, "promoted": False}
    for length in (32, 256):
        if length == 32:
            samples = dev
        else:
            samples = []
            for src, ds, i, _ in selected:
                rgb, gt, valid = ds[i]
                samples.append({"rgb": rgb, "gt": gt, "valid": valid.bool(), "source": src,
                                "scene": q.scene(q.paths(ds, i)[0]), "pairs": q.paths(ds, i)})
            torch.save(samples, args.out/"long_development_data.pt")
        baseline = None
        for key, model in models.items():
            measured, preds = evaluate(model, samples)
            if baseline is None: baseline = measured
            measured["quality_gate"] = quality_gate(baseline["metrics"], measured["metrics"])
            measured["speedup"] = baseline["mean_ms"]/measured["mean_ms"]
            measured["screen_pass"] = key != "dense" and measured["quality_gate"]["pass"] and measured["speedup"] >= 1/.95
            result["L"+str(length)][key] = measured
            torch.save(preds, args.out/("L"+str(length)+"_"+key+"_predictions.pt"))
            q.write(args.out/"results.json", result)
            print(length, key, measured["metrics"]["balanced"], measured["quality_gate"], flush=True)
    for record in protocol["code"]+[protocol["adapter"], protocol["checkpoint"], protocol["refiner"]]:
        assert file_record(record["path"]) == record


if __name__ == "__main__": main()
