"""Training-only threshold provenance, realized schedule and output replay."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import cv2
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qschedule_study import padded,predict
from sokkanaem.protocol import file_record
from sokkanaem.qschedule import ScheduleRisk
from sokkanaem.qflow import gray_image,correspondence,warp_depth
from sokkanaem.qquality import depth_retention


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise FileExistsError("New output required")
    torch.set_num_threads(4);cv2.setNumThreads(1)
    root=Path("work_dirs/qschedule_20260909");pp=json.loads((root/"protocol.json").read_text())
    results=json.loads((root/"results.json").read_text());thresholds=json.loads((root/"thresholds.json").read_text())
    records=pp["code"]+pp["development"]+[pp["checkpoint"],pp["training_cache"],pp["parent_protocol"]]
    for r in records:assert file_record(r["path"])==r
    traces=torch.load(root/"traces.pt",weights_only=False)
    held=[r for r in traces if not r["fit"]];hx,hy,hv=padded(held,"cuda")
    threshold_checks={}
    for mode in ("carry","reset","mlp"):
        model=ScheduleRisk("mlp" if mode=="mlp" else "ssm").cuda().eval()
        model.load_state_dict(torch.load(root/(mode+"_risk.pt"),map_location="cuda",weights_only=False)["risk"],strict=True)
        scores=predict(model,hx,mode=="reset");eligible=hv.bool();eligible[:,0]=False
        values=torch.quantile(scores[eligible],scores.new_tensor([.25,.5,.75])).cpu()
        expected=torch.tensor([thresholds[mode][str(n)] for n in (25,50,75)])
        torch.testing.assert_close(values,expected,rtol=0,atol=0)
        threshold_checks[mode]={"max_abs":float((values-expected).abs().max()),"calibration_extension_observations":int(eligible.sum())}
    del model
    audit={"role":"posthoc_reused_development_schedule_audit_no_threshold_changes", "source":file_record(__file__),
           "threshold_checks":threshold_checks,"inputs":[file_record(root/n) for n in ("protocol.json","thresholds.json","traces.pt","results.json")],"data":{}}
    engine=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    for length,rec in zip(("L32","L256"),pp["development"]):
        samples=torch.load(rec["path"],weights_only=False)
        dense=torch.load(root/(length+"_dense_predictions.pt"),weights_only=False)
        stats={}
        for name,measured in results[length].items():
            if name in ("baseline","output_k2","output_k4"):continue
            path=root/(length+"_"+name+"_predictions.pt");preds=torch.load(path,weights_only=False)
            audit["inputs"].append(file_record(path));reasons=Counter();compared=0;abs_errors=[];accepted=[];rejected=[]
            elapsed=iter(measured["per_frame_ms"]);times={}
            mode=name.split("_")[0];percentile=name.split("_")[1][1:];threshold=thresholds[mode][percentile]
            for ci,(s,tele) in enumerate(zip(samples,measured["telemetry"])):
                age=0
                for t,info in enumerate(tele["frames"]):
                    reason=info["reason"];reasons[reason]+=1;times.setdefault(reason,[]).append(next(elapsed))
                    expected_gap=0 if t==0 else age+1;assert info["gap_attempted"]==expected_gap
                    if t==0 or expected_gap==4:assert info["full_refresh"]
                    elif expected_gap==1:assert not info["full_refresh"] and info["incoming_h_abs_mean"]==0.
                    else:assert info["full_refresh"]==(info["predicted_risk"]>threshold)
                    if t>0 and expected_gap<4:
                        previous=s["rgb"][t-1:t].cuda();current=s["rgb"][t:t+1].cuda()
                        flow,reliable=correspondence(gray_image(previous),gray_image(current),engine)
                        candidate,_=warp_depth(preds[ci][t-1:t].cuda(),flow,reliable,"nearest")
                        if not info["full_refresh"]:
                            assert torch.equal(candidate.cpu(),preds[ci][t:t+1]);compared+=1
                        if expected_gap>=2:
                            depth,edge=depth_retention(candidate,dense[ci][t:t+1].cuda());actual=float(depth+2*edge)
                            abs_errors.append(abs(actual-info["predicted_risk"]))
                            (rejected if info["full_refresh"] else accepted).append(actual)
                    age=0 if info["full_refresh"] else expected_gap;assert info["cache_age"]==age and age<=3
            stats[name]={"reasons":dict(reasons),"skipped_output_exact_replay_frames":compared,
                         "extension_risk_mae":statistics.mean(abs_errors) if abs_errors else None,
                         "accepted_true_teacher_risk_mean":statistics.mean(accepted) if accepted else None,
                         "rejected_true_teacher_risk_mean":statistics.mean(rejected) if rejected else None,
                         "reason_mean_ms":{n:statistics.mean(v) for n,v in times.items()},
                         "extra_q0_calls_saved_vs_k2":results[length]["output_k2"]["full_backbone_calls"]-measured["full_backbone_calls"],
                         "latency_change_vs_k2_pct":100*(measured["mean_ms"]/results[length]["output_k2"]["mean_ms"]-1)}
            print(length,name,stats[name],flush=True)
        original=results[length]["carry_q50"];reset=results[length]["carry_q50_h_reset"]
        changed=sum(a["full_refresh"]!=b["full_refresh"] for ta,tb in zip(original["telemetry"],reset["telemetry"]) for a,b in zip(ta["frames"],tb["frames"]))
        audit["data"][length]={"policies":stats,"same_weight_h_removal_changed_refresh_decisions":changed}
    audit["notes"]=["True risk here means actual Q0 logdepth+2loggradient proxy, NOT GT risk or a quality guarantee.",
                    "Timing is original synchronized run grouped posthoc; individual reasons have different frames and counts, not matched causal component benchmarks.",
                    "Removing h changes future schedules and candidate caches too; whole-policy intervention, not isolated per-frame memory quality."]
    args.out.mkdir(parents=True);q.write(args.out/"audit.json",audit)
    for r in records:assert file_record(r["path"])==r


if __name__=="__main__":main()
