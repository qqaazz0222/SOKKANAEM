"""Matched RGB-only risk fitting, early/late decision cost ablation."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact
from scripts.qschedule_study import padded,predict
from scripts.qstream_study import evaluate,quality_gate
from sokkanaem.protocol import file_record
from sokkanaem.qschedule import ScheduleRisk
from sokkanaem.qpreflow import PreflowScheduledFlow
from sokkanaem.qflow import RGBFlowStream
from sokkanaem.qrecurrent_flow import K4OutputFlow


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise FileExistsError("New output required")
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    parent=Path("work_dirs/qschedule_20260909");pp=json.loads((parent/"protocol.json").read_text())
    for r in pp["code"]+pp["development"]+[pp["checkpoint"],pp["training_cache"]]:assert file_record(r["path"])==r
    exact=load_exact(pp["checkpoint"])
    traces=torch.load(parent/"traces.pt",weights_only=False)
    # Same labels and split as previous20-feature study; remove only flow inputs.
    for row in traces:row["x"][:,14:]=0
    fit=[r for r in traces if r["fit"]];held=[r for r in traces if not r["fit"]]
    fitpaths={p for r in fit for pair in r["pairs"] for p in pair};heldpaths={p for r in held for pair in r["pairs"] for p in pair}
    assert not fitpaths&heldpaths
    x,y,valid=padded(fit,"cuda");hx,hy,hv=padded(held,"cuda")
    raw=torch.cat([r["x"] for r in fit]).cuda();center=raw.mean(0);scale=raw.std(0,unbiased=False).clamp_min(1e-4)
    order=json.loads((parent/"training_order.json").read_text())
    args.out.mkdir(parents=True)
    protocol={"role":"RGB_only_preflow_policy_reused_development_no_final", "checkpoint":pp["checkpoint"],"development":pp["development"],
              "parent_protocol":file_record(parent/"protocol.json"),"traces":file_record(parent/"traces.pt"),"order":file_record(parent/"training_order.json"),
              "training":"same fit144/calibration48clips; same labels, seed741,1000steps,batch32,lr.001, initialization and exact sampled order as20input predecessor",
              "features":"only first14 RGB/anchor/age stats; last6 inputs zero, parameters retained for architectural control but96input weights have zero data gradient",
              "thresholds":"model-specific calibration prediction quantiles .25/.5/.75 on gap>=2; before dev, unchanged operating-rule",
              "ablation":"carry/reset/MLP early decisions; same carry weights+thresholds run late too, output/schedule parity required",
              "limits":"risk target still Q0 depth+2gradient, not GT boundary/flat risk; loss unchanged to isolate cost/input removal; h resets on fullQ0",
              "timing":"includes grayprep/RGBstats/policy/flow when run/Q0/transfers; synchronized GPUresident excludes initialIO/H2D; no edge claim",
              "code":[file_record(p) for p in [__file__,"sokkanaem/qpreflow.py","sokkanaem/qschedule.py","sokkanaem/qrisk.py","sokkanaem/qflow.py",
                        "sokkanaem/ssm.py","sokkanaem/qrecurrent_flow.py","sokkanaem/qstream.py","sokkanaem/qmodel.py","scripts/qschedule_study.py",
                        "scripts/qquality_study.py","scripts/qstream_study.py","scripts/quality_refinement.py","sokkanaem/sharpness.py"]]}
    q.write(args.out/"protocol.json",protocol);models={};thresholds={};diagnostics={}
    for mode in ("carry","reset","mlp"):
        torch.manual_seed(741);model=ScheduleRisk("mlp" if mode=="mlp" else "ssm").cuda()
        model.center.copy_(center);model.scale.copy_(scale)
        opt=torch.optim.AdamW(model.parameters(),lr=.001);history=[]
        for step,idx in enumerate(order,1):
            scores=predict(model,x[idx],mode=="reset")
            loss=(F.smooth_l1_loss(scores/.05,y[idx]/.05,reduction="none")*valid[idx]).sum()/valid[idx].sum()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step==1 or step%200==0:history.append({"step":step,"loss":float(loss.detach())});print(mode,history[-1],flush=True)
        model.eval().requires_grad_(False);models[mode]=model
        torch.save({"risk":model.state_dict(),"mode":mode},args.out/(mode+"_risk.pt"));q.write(args.out/(mode+"_history.json"),history)
        with torch.no_grad():
            hp=predict(model,hx,mode=="reset");eligible=hv.bool();eligible[:,0]=False
            values=torch.quantile(hp[eligible],hp.new_tensor([.25,.5,.75])).cpu().tolist()
            thresholds[mode]={str(n):v for n,v in zip((25,50,75),values)}
            diagnostics[mode]={"fit_mae":float(((predict(model,x,mode=="reset")-y).abs()*valid).sum()/valid.sum()),
                               "calibration_mae":float(((hp-hy).abs()*hv).sum()/hv.sum())}
    q.write(args.out/"thresholds.json",thresholds);q.write(args.out/"risk_diagnostics.json",diagnostics)
    print("thresholds",thresholds,"diagnostics",diagnostics,flush=True)
    result={"promoted":False};parity=[]
    for length,rec in zip(("L32","L256"),pp["development"]):
        samples=torch.load(rec["path"],weights_only=False);baseline,dense=evaluate(exact,samples)
        result[length]={"baseline":baseline};torch.save(dense,args.out/(length+"_dense_predictions.pt"))
        wrappers={"output_k2":RGBFlowStream(exact),"output_k4":K4OutputFlow(exact)}
        for mode,model in models.items():
            for percentile,threshold in thresholds[mode].items():wrappers[mode+"_q"+percentile]=PreflowScheduledFlow(exact,model,threshold,reset_h=mode=="reset")
        for percentile,threshold in thresholds["carry"].items():wrappers["carry_q"+percentile+"_late"]=PreflowScheduledFlow(exact,models["carry"],threshold,late=True)
        for name,wrapper in wrappers.items():
            measured,preds=evaluate(wrapper,samples)
            measured["quality_gate"]=quality_gate(baseline["metrics"],measured["metrics"])
            measured["speedup"]=baseline["mean_ms"]/measured["mean_ms"]
            measured["screen_pass"]=measured["quality_gate"]["pass"] and measured["speedup"]>=1/.95
            if name!="output_k2":measured["speedup_vs_k2"]=result[length]["output_k2"]["mean_ms"]/measured["mean_ms"]
            full=0
            for ci,tele in enumerate(measured["telemetry"]):
                for t,info in enumerate(tele["frames"]):
                    if info["full_refresh"]:assert torch.equal(preds[ci][t],dense[ci][t]);full+=1
                    if info.get("reason")=="risk" and not name.endswith("_late"):assert info["flow_calls"]==0
            row={"length":length,"policy":name,"exact_refresh_comparisons":full}
            if name.endswith("_late"):
                early=name[:-5];ep=torch.load(args.out/(length+"_"+early+"_predictions.pt"),weights_only=False)
                assert all(torch.equal(a,b) for a,b in zip(ep,preds))
                for a,b in zip(result[length][early]["telemetry"],measured["telemetry"]):
                    assert [f["full_refresh"] for f in a["frames"]]==[f["full_refresh"] for f in b["frames"]]
                row["early_late_prediction_parity_frames"]=sum(len(p) for p in preds)
            parity.append(row);result[length][name]=measured
            torch.save(preds,args.out/(length+"_"+name+"_predictions.pt"));q.write(args.out/"results.json",result);q.write(args.out/"parity.json",parity)
            print(length,name,"ms",measured["mean_ms"],"skip",measured["skip_fraction"],"gate",measured["quality_gate"],flush=True)
    for r in protocol["code"]+protocol["development"]+[protocol["checkpoint"],protocol["parent_protocol"],protocol["traces"],protocol["order"]]:assert file_record(r["path"])==r
    q.write(args.out/"integrity.json",{"hashes_unchanged":True,"full_refresh_comparisons":sum(r["exact_refresh_comparisons"] for r in parity),
                                      "early_late_prediction_parity_frames":sum(r.get("early_late_prediction_parity_frames",0) for r in parity)})


if __name__=="__main__":main()
