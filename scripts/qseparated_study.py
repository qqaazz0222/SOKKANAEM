"""Matched post-training ablations: hold CLS, pooled calibration, both, neither."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact
from scripts.qstream_study import evaluate, quality_gate
from sokkanaem.protocol import file_record
from sokkanaem.qflat_state import PersistentStream
from sokkanaem.qseparated import SeparatedDelta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    parent = Path("work_dirs/qflat_carry_flat_20260908")
    pp = json.loads((parent/"protocol.json").read_text())
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    exact = load_exact(pp["checkpoint"])
    weights = torch.load(parent/"adapter.pt", map_location="cuda", weights_only=False)["adapter"]
    paths = [Path("work_dirs/qstream_20260908/development_data.pt"), Path("work_dirs/qquality_audit_20260908/long_development_data.pt")]
    protocol = {"role": "posttraining_development_diagnostic_no_retraining_no_promotion", "checkpoint": pp["checkpoint"],
                "adapter": file_record(parent/"adapter.pt"), "development": [file_record(p) for p in paths],
                "policy": "K2 with same conservative refresh h carry; same trained patch updates",
                "variants": ["unchanged", "hold_cls", "hold_pooled", "hold_both"],
                "limits": "holding pooled fixes its delta, not all metric effects: normalized decoder disparity still changes; prior opened development only",
                "code": [file_record(p) for p in [__file__, "sokkanaem/qseparated.py", "sokkanaem/qflat_state.py",
                          "sokkanaem/qdelta_ssm.py", "scripts/qquality_study.py", "scripts/qstream_study.py",
                          "sokkanaem/qstream.py", "sokkanaem/sharpness.py", "scripts/quality_refinement.py"]]}
    args.out.mkdir(parents=True); q.write(args.out/"protocol.json", protocol)
    result = {"promoted": False}
    for length, path in zip(("L32", "L256"), paths):
        samples = torch.load(path, weights_only=False)
        baseline, _ = evaluate(exact, samples); result[length] = {"baseline": baseline}
        for name, cls, pooled in (("unchanged", False, False), ("hold_cls", True, False),
                                   ("hold_pooled", False, True), ("hold_both", True, True)):
            adapter = SeparatedDelta(cls, pooled).cuda().eval(); adapter.load_state_dict(weights, strict=True)
            measured, preds = evaluate(PersistentStream(exact, adapter, carry=True), samples)
            measured["quality_gate"] = quality_gate(baseline["metrics"], measured["metrics"])
            measured["speedup"] = baseline["mean_ms"]/measured["mean_ms"]
            measured["screen_pass"] = measured["quality_gate"]["pass"] and measured["speedup"] >= 1/.95
            result[length][name] = measured
            torch.save(preds, args.out/(length+"_"+name+"_predictions.pt")); q.write(args.out/"results.json", result)
            print(length, name, measured["metrics"]["balanced"], measured["quality_gate"], flush=True)
    for r in protocol["code"]+[protocol["checkpoint"], protocol["adapter"]]: assert file_record(r["path"]) == r


if __name__ == "__main__": main()
