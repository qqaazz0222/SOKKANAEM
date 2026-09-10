"""Predeclared training-free RGB-correspondence controls on reused development."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact
from scripts.qstream_study import evaluate, quality_gate
from sokkanaem.protocol import file_record, runtime_versions
from sokkanaem.qstream import ChangeAwareQ0
from sokkanaem.qflow import RGBFlowStream


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.manual_seed(737)
    torch.backends.cudnn.benchmark = False
    pp = json.loads(Path("work_dirs/qquality_feature_20260908/protocol.json").read_text())
    exact = load_exact(pp["checkpoint"])
    paths = [Path("work_dirs/qstream_20260908/development_data.pt"),
             Path("work_dirs/qquality_audit_20260908/long_development_data.pt")]
    policies = {"flow_bilinear": {"mode": "bilinear"}, "flow_nearest": {"mode": "nearest"},
                "flow_gated": {"mode": "nearest", "gated": True},
                "flow_safe": {"mode": "nearest", "gated": True, "max_unreliable": .1}}
    args.out.mkdir(parents=True)
    protocol = {"role": "reused_development_no_training_no_final_no_promotion", "checkpoint": pp["checkpoint"],
                "data": [file_record(p) for p in paths], "policies": policies,
                "flow": "OpenCV DIS ULTRAFAST, grayscale128, forward and backward; FB<=1.5pixels, photometric<=12.75/255, bounds and 3x3 erosion",
                "confidence": "uncalibrated heuristic, NOT true occlusion or guaranteed correctness",
                "schedule": "K2 output controls, no selective SSM; confidence rejection resets period with full Q0",
                "timing": "synchronized per-frame wall time, GPU-resident input; includes gray resize, GPU->CPU, two CPU flows, CPU->GPU, warp, confidence, Q0; excludes initial IO/H2D",
                "thresholds": "fixed before this run, no selection on final; unchanged Q0 quality gate and >=5pct mean latency reduction",
                "runtime": runtime_versions(), "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
                "gpu": torch.cuda.get_device_name(),
                "code": [file_record(p) for p in [__file__, "sokkanaem/qflow.py", "scripts/qstream_study.py",
                          "scripts/qquality_study.py", "sokkanaem/qstream.py", "sokkanaem/qmodel.py",
                          "scripts/quality_refinement.py", "sokkanaem/sharpness.py"]]}
    q.write(args.out/"protocol.json", protocol)
    results = {"promoted": False}; parity = []
    for length, path in zip(("L32", "L256"), paths):
        samples = torch.load(path, weights_only=False)
        baseline, dense = evaluate(exact, samples)
        results[length] = {"baseline": baseline}
        torch.save(dense, args.out/(length+"_dense_predictions.pt"))
        models = {"hold_k2": ChangeAwareQ0(exact, refresh_every=2, periodic_only=True)}
        models.update({name: RGBFlowStream(exact, **config) for name, config in policies.items()})
        for name, model in models.items():
            measured, predictions = evaluate(model, samples)
            measured["quality_gate"] = quality_gate(baseline["metrics"], measured["metrics"])
            measured["speedup"] = baseline["mean_ms"]/measured["mean_ms"]
            measured["screen_pass"] = measured["quality_gate"]["pass"] and measured["speedup"] >= 1/.95
            full = 0; maxerr = 0.
            for ci, telemetry in enumerate(measured["telemetry"]):
                for t, info in enumerate(telemetry["frames"]):
                    if info["full_refresh"]:
                        err = float((predictions[ci][t]-dense[ci][t]).abs().max())
                        assert err == 0., (length, name, ci, t, err)
                        full += 1; maxerr = max(maxerr, err)
            parity.append({"length": length, "policy": name, "full_refresh_frames": full, "max_abs": maxerr})
            results[length][name] = measured
            torch.save(predictions, args.out/(length+"_"+name+"_predictions.pt"))
            q.write(args.out/"results.json", results); q.write(args.out/"parity.json", parity)
            print(length, name, measured["metrics"]["balanced"], "ms", measured["mean_ms"],
                  "skip", measured["skip_fraction"], "gate", measured["quality_gate"], flush=True)
    for r in protocol["code"]+protocol["data"]+[protocol["checkpoint"]]:
        assert file_record(r["path"]) == r
    q.write(args.out/"integrity.json", {"input_hashes_unchanged": True, "full_refresh_parity_zero": True,
                                      "full_refresh_frames": sum(r["full_refresh_frames"] for r in parity)})


if __name__ == "__main__": main()
