"""Read-only experiment audit; writes a new consolidated result directory."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record
from scripts import quality_refinement as q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    torch.set_num_threads(4)
    runs = [Path("work_dirs/qflat_"+name+"_20260908") for name in
            ("reset_control", "reset_flat", "carry_control", "carry_flat")]
    dense = {l: torch.load("work_dirs/qquality_audit_20260908/"+l+"_dense_predictions.pt", weights_only=False)
             for l in ("L32", "L256")}
    report = {"promoted": False, "source": file_record(__file__), "runs": [], "parity": [], "refresh_frames_checked": 0}
    for run in runs:
        protocol = json.loads((run/"protocol.json").read_text())
        results = json.loads((run/"results.json").read_text())
        for record in protocol["code"]+[protocol["checkpoint"], protocol["initial_adapter"], protocol["train_cache"], protocol["training_gt"]]+protocol["dev"]:
            assert file_record(record["path"]) == record
        for length in ("L32", "L256"):
            base = results[length]["baseline"]["metrics"]["balanced"]
            for mode in ("reset", "carry"):
                result = results[length][mode]; m = result["metrics"]["balanced"]
                predictions = torch.load(run/(length+"_"+mode+"_predictions.pt"), weights_only=False)
                count = 0; maxerr = 0.
                for p, target, clip in zip(predictions, dense[length], result["telemetry"]):
                    idx = torch.tensor([i for i, f in enumerate(clip["frames"]) if f["full_refresh"]])
                    maxerr = max(maxerr, float((p[idx]-target[idx]).abs().max()))
                    torch.testing.assert_close(p[idx], target[idx], rtol=0, atol=0)
                    count += len(idx)
                assert count == result["full_backbone_calls"] and count*2 == result["frames"]
                report["refresh_frames_checked"] += count
                report["parity"].append({"run": str(run), "length": length, "mode": mode, "frames": count, "max_abs": maxerr})
                report["runs"].append({"run": str(run), "trained_carry": protocol["carry"], "flat_loss": protocol["flat"],
                                       "length": length, "inference": mode, "raw": m["absrel_raw"], "f1": m["boundary_f1"],
                                       "flat_tv": m["flat_tv"], "flat_tv_relative_pct": 100*(m["flat_tv"]/base["flat_tv"]-1),
                                       "mean_ms": result["mean_ms"], "speedup": result["speedup"],
                                       "quality": result["quality_gate"], "joint_pass": result["screen_pass"]})
    args.out.mkdir(parents=True); q.write(args.out/"audit.json", report)
    print("exact refresh outputs", report["refresh_frames_checked"], flush=True)
    for row in report["runs"]:
        print(row["length"], row["trained_carry"], row["flat_loss"], row["inference"],
              "raw", round(row["raw"], 6), "F1", round(row["f1"], 6), "flat%", round(row["flat_tv_relative_pct"], 3),
              "ms", round(row["mean_ms"], 3), "pass", row["joint_pass"], flush=True)


if __name__ == "__main__": main()
