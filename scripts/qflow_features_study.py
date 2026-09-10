"""Same-flow feature transport x frozen SSM update 2x2, no new training."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact, MotionStream
from scripts.qstream_study import evaluate, quality_gate
from sokkanaem.protocol import file_record
from sokkanaem.qseparated import SeparatedDelta
from sokkanaem.qflow import RGBFlowStream
from sokkanaem.qflow_features import RGBFlowFeatureStream


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    root = Path("work_dirs/qquality_feature_20260908")
    pp = json.loads((root/"protocol.json").read_text())
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.backends.cudnn.benchmark = False
    exact = load_exact(pp["checkpoint"])
    adapter = SeparatedDelta(True, True).cuda().eval().requires_grad_(False)
    adapter.load_state_dict(torch.load(root/"adapter.pt", map_location="cuda", weights_only=False)["adapter"], strict=True)
    paths = [Path("work_dirs/qstream_20260908/development_data.pt"), Path("work_dirs/qquality_audit_20260908/long_development_data.pt")]
    args.out.mkdir(parents=True)
    protocol = {"role": "reused_development_frozen_adapter_architectural_transfer_not_final", "checkpoint": pp["checkpoint"],
                "adapter": file_record(root/"adapter.pt"), "data": [file_record(p) for p in paths],
                "experiment": "nearest/bilinear feature+h warp x SSM update on/off; same weights, same original RGB-change mask; output-nearest and unwarped-SSM controls",
                "training": "none; adapter trained without this DIS alignment, distribution shift is a limitation",
                "state": "K2 full refresh resets h; transporting h is implemented but h is zero before each skip; not a recurrent memory contribution",
                "globals": "CLS and raw pooled held, no depth blending or global metric delta",
                "timing": "same synchronized GPU-resident evaluator; includes two CPU flows, all transfers, feature+h warp, SSM and decoder; no edge deployment claim",
                "code": [file_record(p) for p in [__file__, "sokkanaem/qflow_features.py", "sokkanaem/qflow.py", "sokkanaem/qseparated.py",
                          "sokkanaem/qdelta_ssm.py", "scripts/qquality_study.py", "scripts/qstream_study.py", "sokkanaem/qstream.py",
                          "sokkanaem/qmodel.py", "scripts/quality_refinement.py", "sokkanaem/sharpness.py"]]}
    q.write(args.out/"protocol.json", protocol); result = {"promoted": False}; parity = []
    for length, path in zip(("L32", "L256"), paths):
        samples = torch.load(path, weights_only=False)
        baseline, dense = evaluate(exact, samples); result[length] = {"baseline": baseline}
        models = {"output_nearest": RGBFlowStream(exact), "unwarped_ssm": MotionStream(exact, adapter, refresh_every=2)}
        for mode in ("nearest", "bilinear"):
            for disabled in (True, False):
                models[mode+("_no_ssm" if disabled else "_ssm")] = RGBFlowFeatureStream(exact, adapter, mode, disabled)
        for name, model in models.items():
            measured, predictions = evaluate(model, samples)
            measured["quality_gate"] = quality_gate(baseline["metrics"], measured["metrics"])
            measured["speedup"] = baseline["mean_ms"]/measured["mean_ms"]
            measured["screen_pass"] = measured["quality_gate"]["pass"] and measured["speedup"] >= 1/.95
            full = 0
            for ci, tele in enumerate(measured["telemetry"]):
                for t, info in enumerate(tele["frames"]):
                    if info["full_refresh"]:
                        assert torch.equal(predictions[ci][t], dense[ci][t]); full += 1
            parity.append({"length": length, "model": name, "full_refresh_frames": full, "max_abs": 0.})
            result[length][name] = measured
            torch.save(predictions, args.out/(length+"_"+name+"_predictions.pt"))
            q.write(args.out/"results.json", result); q.write(args.out/"parity.json", parity)
            print(length, name, measured["metrics"]["balanced"], "ms", measured["mean_ms"], measured["quality_gate"], flush=True)
    for r in protocol["code"]+protocol["data"]+[protocol["checkpoint"], protocol["adapter"]]: assert file_record(r["path"]) == r
    q.write(args.out/"integrity.json", {"hashes_unchanged": True, "full_refresh_parity_zero_frames": sum(r["full_refresh_frames"] for r in parity)})


if __name__ == "__main__": main()
