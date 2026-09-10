"""Saved-prediction memory intervention and Q0 temporal-delta diagnostics."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from sokkanaem.protocol import file_record


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    torch.set_num_threads(4)
    root=Path("work_dirs/qrecurrent_20260909")
    pp=json.loads((root/"protocol.json").read_text()); results=json.loads((root/"results.json").read_text())
    records=pp["code"]+pp["development"]+[pp["checkpoint"],pp["training_data"],json.loads((root/"cache_record.json").read_text())]
    for r in records: assert file_record(r["path"])==r
    gradient={}
    for name in ("carry","reset","mlp"):
        rows=json.loads((root/(name+"_history.json")).read_text())
        values=[r["A_data_gradient_max"] for r in rows if r["A_data_gradient_max"] is not None]
        gradient[name]={"logged_steps":len(rows),"A_gradient_max":max(values) if values else None,
                        "steps_with_nonzero_A_gradient":sum(v>0 for v in values)}
    assert gradient["carry"]["A_gradient_max"]>0 and gradient["reset"]["A_gradient_max"]==0
    audit={"role":"posthoc_reused_development_memory_and_teacher_temporal_diagnostic_not_final",
           "source":file_record(__file__),"training_gradient":gradient,"data":{},"inputs":[]}
    for length,rec in zip(("L32","L256"),pp["development"]):
        samples=torch.load(rec["path"],weights_only=False)
        dense_path=root/(length+"_dense_predictions.pt"); dense=torch.load(dense_path,weights_only=False)
        audit["inputs"].append(file_record(dense_path))
        temporal={}; incoming={}; interventions={}
        for name,measured in results[length].items():
            if name=="baseline":continue
            path=root/(length+"_"+name+"_predictions.pt")
            preds=torch.load(path,weights_only=False); audit["inputs"].append(file_record(path))
            rows=[]
            for s,p,d in zip(samples,preds,dense):
                difference=torch.diff(p.clamp_min(1e-4).log(),dim=0)-torch.diff(d.clamp_min(1e-4).log(),dim=0)
                rows.append({"source":s["source"],"scene":s["scene"],"metrics":{"q0_log_temporal_delta_l1":float(difference.abs().mean())}})
            temporal[name]=q.aggregate(rows)
            if "incoming_h_abs_mean" in measured["telemetry"][0]["frames"][0]:
                incoming[name]={str(age):statistics.mean(f["incoming_h_abs_mean"] for t in measured["telemetry"] for f in t["frames"] if f["cache_age"]==age)
                                for age in (0,1,2,3)}
            del preds
        for trained in ("carry","reset"):
            a=torch.load(root/(length+"_"+trained+"_carry_predictions.pt"),weights_only=False)
            b=torch.load(root/(length+"_"+trained+"_reset_predictions.pt"),weights_only=False)
            count=0; maxerr=0.; rows=[]
            for s,p,r in zip(samples,a,b):
                for offset in (0,1):
                    assert torch.equal(p[offset::4],r[offset::4]); count+=len(p[offset::4])
                late=torch.arange(len(p))%4>=2
                maxerr=max(maxerr,float((p[late]-r[late]).abs().max()))
                rows.append({"source":s["source"],"scene":s["scene"],"metrics":{"later_updates_same_weights_log_difference":float((p[late].log()-r[late].log()).abs().mean())}})
            interventions[trained]={"identical_refresh_and_first_update_comparisons":count,"later_max_abs_depth_difference":maxerr,
                                    "later_difference":q.aggregate(rows),
                                    "note":"same checkpoint, only h zeroing changed; feature cache feedback remains in both; difference is effect, not proof of benefit"}
        audit["data"][length]={"incoming_h_by_age":incoming,"same_weight_interventions":interventions,"temporal":temporal}
        print(length,"memory",interventions,flush=True)
        print(length,"Q0 temporal delta",{k:v["balanced"] for k,v in temporal.items()},flush=True)
    audit["temporal_note"]="Mean absolute error in consecutive log-depth differences relative to denseQ0, all pixels, source/scene balanced. Not GT temporal accuracy, OPW or paper TCE; no optical-flow alignment; no new gate."
    args.out.mkdir(parents=True); q.write(args.out/"audit.json",audit)
    for r in records:assert file_record(r["path"])==r


if __name__=="__main__":main()
