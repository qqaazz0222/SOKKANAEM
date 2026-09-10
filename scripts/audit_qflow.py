"""Post-screen phase sensitivity and three alternating timing repeats."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import cv2
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact
from scripts.qquality_audit import latency_audit
from scripts.qstream_study import evaluate, quality_gate
from sokkanaem.protocol import file_record
from sokkanaem.qflow import RGBFlowStream


class ShiftedFlow(RGBFlowStream):
    @torch.no_grad()
    def step(self, frame, state=None):
        n = 0 if state is None else state["n"]
        if n == 1:
            state = {**state, "age": 1}
        pred, next_state, info = super().step(frame, state)
        next_state["n"] = n+1
        return pred, next_state, info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    root = Path("work_dirs/qflow_20260909")
    pp = json.loads((root/"protocol.json").read_text()); measured = json.loads((root/"results.json").read_text())
    for r in pp["code"]+pp["data"]+[pp["checkpoint"]]: assert file_record(r["path"]) == r
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.backends.cudnn.benchmark = False
    exact = load_exact(pp["checkpoint"])
    args.out.mkdir(parents=True)
    protocol = {"role": "post_selection_reused_development_audit_not_independent_validation",
                "parent": file_record(root/"protocol.json"), "results": file_record(root/"results.json"),
                "source": file_record(__file__), "timing_source": file_record("scripts/qquality_audit.py"),
                "timing": "3 alternating repeats,64warmup; synchronized per-frame GPU-resident, flow/transfers included; shared GPU not device benchmark",
                "phase": "first two frames exact then K2; no thresholds changed"}
    q.write(args.out/"protocol.json", protocol); result = {}
    for length, rec in zip(("L32", "L256"), pp["data"]):
        samples = torch.load(rec["path"], weights_only=False)
        dense = torch.load(root/(length+"_dense_predictions.pt"), weights_only=False)
        phase, predictions = evaluate(ShiftedFlow(exact), samples)
        phase["quality_gate"] = quality_gate(measured[length]["baseline"]["metrics"], phase["metrics"])
        count = 0
        for ci, tele in enumerate(phase["telemetry"]):
            for t, info in enumerate(tele["frames"]):
                if info["full_refresh"]:
                    assert torch.equal(predictions[ci][t], dense[ci][t]); count += 1
        phase["full_refresh_parity_zero_frames"] = count
        torch.save(predictions, args.out/(length+"_shifted_predictions.pt"))
        print(length, "shifted", phase["metrics"]["balanced"], phase["quality_gate"], flush=True)
        times = latency_audit({"dense": exact, "flow_nearest": RGBFlowStream(exact)}, samples)
        means = {k: statistics.mean(r["mean_ms"] for r in rows) for k, rows in times.items()}
        summary = {"mean_ms": means, "speedup_ratio_of_means": means["dense"]/means["flow_nearest"],
                   "latency_reduction_pct": 100*(1-means["flow_nearest"]/means["dense"]),
                   "repeat_min_max_ms": {k: [min(r["mean_ms"] for r in rows), max(r["mean_ms"] for r in rows)] for k,rows in times.items()}}
        result[length] = {"shifted_phase": phase, "timings": times, "timing_summary": summary}
        q.write(args.out/"audit.json", result); print(length, summary, flush=True)
    for r in pp["code"]+pp["data"]+[pp["checkpoint"], protocol["source"], protocol["timing_source"]]:
        assert file_record(r["path"]) == r
    q.write(args.out/"integrity.json", {"input_hashes_unchanged": True, "shifted_full_refresh_parity_zero": True})


if __name__ == "__main__": main()
