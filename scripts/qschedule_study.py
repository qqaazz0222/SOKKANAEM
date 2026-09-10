"""Train-only calibrated risk percentile scheduling; no development tuning."""
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
from scripts.qstream_study import evaluate,quality_gate
from sokkanaem.protocol import file_record,FINAL_TEST_SEQUENCES,runtime_versions
from sokkanaem.qquality import depth_retention
from sokkanaem.qflow import warp_depth,RGBFlowStream
from sokkanaem.qrecurrent_flow import K4OutputFlow
from sokkanaem.qschedule import scheduling_features,ScheduleRisk,ScheduledFlow


@torch.no_grad()
def make_traces(train):
    traces=[];seen={};fit_paths=set();held_paths=set()
    for ci,s in enumerate(train):
        src=s["source"];idx=seen.get(src,0);seen[src]=idx+1;fit=idx<48
        (fit_paths if fit else held_paths).update(str(Path(p).resolve()) for pair in s["pairs"] for p in pair)
        rgb=s["rgb"].cuda();teacher=s["teacher"].cuda()
        # Every possible full-refresh anchor with an extension decision in a4frame clip.
        for anchor in (0,1):
            previous=teacher[anchor:anchor+1];xx=[];yy=[]
            for t in range(anchor+1,4):
                moved,reliable=warp_depth(torch.cat((previous,rgb[t-1:t]),1),s["flows"][t-1],s["reliable"][t-1],"nearest")
                previous=moved[:,:1]
                xx.append(scheduling_features(rgb[t:t+1],rgb[anchor:anchor+1],t-anchor,previous,moved[:,1:],reliable,s["flows"][t-1]).cpu())
                depth,edge=depth_retention(previous,teacher[t:t+1]);yy.append((depth+2*edge).reshape(1,1).cpu())
            traces.append({"x":torch.cat(xx),"y":torch.cat(yy),"fit":fit,"source":src,"clip":ci,"anchor":anchor,"pairs":s["pairs"]})
    assert seen=={"tum":64,"bonn":64,"vkitti2":64},seen
    assert not fit_paths&held_paths
    return traces


def padded(traces,device):
    x=torch.zeros(len(traces),3,20,device=device);y=torch.zeros(len(traces),3,1,device=device)
    valid=torch.zeros(len(traces),3,1,device=device)
    for i,row in enumerate(traces):
        n=len(row["x"]);x[i,:n]=row["x"].to(device);y[i,:n]=row["y"].to(device);valid[i,:n]=1
    return x,y,valid


def predict(model,x,reset):
    h=None;outputs=[]
    for t in range(3):
        p,h=model.step(x[:,t],h,reset);outputs.append(p)
    return torch.stack(outputs,1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise FileExistsError("New output required")
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    parent=Path("work_dirs/qrecurrent_20260909");pp=json.loads((parent/"protocol.json").read_text())
    cache_rec=json.loads((parent/"cache_record.json").read_text());assert file_record(cache_rec["path"])==cache_rec
    exact=load_exact(pp["checkpoint"]);train=torch.load(cache_rec["path"],weights_only=False)
    paths=[Path("work_dirs/qstream_20260908/development_data.pt"),Path("work_dirs/qquality_audit_20260908/long_development_data.pt")]
    dev={n:torch.load(p,weights_only=False) for n,p in zip(("L32","L256"),paths)}
    tf={str(Path(p).resolve()) for s in train for pair in s["pairs"] for p in pair}
    df={str(Path(p).resolve()) for samples in dev.values() for s in samples for pair in s["pairs"] for p in pair}
    assert not tf&df and not any(set(Path(p).parts)&set(FINAL_TEST_SEQUENCES) for p in tf|df)
    args.out.mkdir(parents=True)
    protocol={"role":"new_policy_training_and_train_only_percentiles_reused_development_no_final", "checkpoint":pp["checkpoint"],
              "training_cache":cache_rec,"parent_protocol":file_record(parent/"protocol.json"),"development":[file_record(p) for p in paths],
              "fit":"first48clips/source=144; calibration last16/source=48, disjoint frame paths; not independent new scenes/Q0 training data",
              "traces":"anchors0,1 in each4frame trainingclip; allskip prefixes cover gap1/2/3 without using future inputs; labels depth+2gradient Q0 error",
              "policy":"alwaysskip gap1, learned gap2/3, forcefull gap4; h updated eachskip attempt and reset afterfull; outputnever corrected",
              "features":"14 RGB/anchor/age statistics + uncertainfraction/mean&p95flow/photometricresidual/logdepthgradient/logdepthstd",
              "steps_each":1000,"batch":32,"lr":.001,"seed":741,"models":["carry_ssm","reset_ssm","mlp"],
              "threshold_rule":"per model .25/.50/.75 quantiles of predicted risk on gap>=2 calibration traces; same nominal calibration coverage, thresholds differ, no dev threshold tuning",
              "calibration_warning":"percentile operating points, NOT calibrated probabilities/error bounds; quality gate evaluated independently",
              "gate":"unchanged quality gate +>=5pct latency reduction vsdense; also compare stronger passing output-flow K2",
              "timing":"synchronized GPUresident incl CPUflow/transfers/risk/telemetry/Q0; rejected reuse still pays flow cost; excludes initialIO/H2D; RTX4090 not edge",
              "runtime":runtime_versions(),"opencv":cv2.__version__,
              "code":[file_record(p) for p in [__file__,"sokkanaem/qschedule.py","sokkanaem/qrisk.py","sokkanaem/qflow.py","sokkanaem/qquality.py","sokkanaem/ssm.py",
                        "sokkanaem/qrecurrent_flow.py","sokkanaem/qstream.py","sokkanaem/qmodel.py","scripts/qquality_study.py","scripts/qstream_study.py","scripts/quality_refinement.py","sokkanaem/sharpness.py"]]}
    q.write(args.out/"protocol.json",protocol)
    traces=make_traces(train);del train;torch.save(traces,args.out/"traces.pt")
    fit=[r for r in traces if r["fit"]];held=[r for r in traces if not r["fit"]]
    x,y,valid=padded(fit,"cuda");hx,hy,hv=padded(held,"cuda")
    raw=torch.cat([r["x"] for r in fit]).cuda();center=raw.mean(0);scale=raw.std(0,unbiased=False).clamp_min(1e-4)
    models={};thresholds={};diagnostics={};initial=None
    order=torch.randint(len(fit),(1000,32),generator=torch.Generator().manual_seed(741)).tolist()
    q.write(args.out/"training_order.json",order)
    for mode in ("carry","reset","mlp"):
        torch.manual_seed(741);model=ScheduleRisk("mlp" if mode=="mlp" else "ssm").cuda()
        model.center.copy_(center);model.scale.copy_(scale)
        common={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith(("input.","head."))}
        if initial is None:initial=common
        else:assert all(torch.equal(p,initial[n]) for n,p in common.items())
        opt=torch.optim.AdamW(model.parameters(),lr=.001);history=[]
        for step,idx in enumerate(order,1):
            prediction=predict(model,x[idx],mode=="reset")
            loss=(F.smooth_l1_loss(prediction/.05,y[idx]/.05,reduction="none")*valid[idx]).sum()/valid[idx].sum()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step==1 or step%200==0:
                history.append({"step":step,"loss":float(loss.detach())});print(mode,history[-1],flush=True)
        model.eval().requires_grad_(False);models[mode]=model
        q.write(args.out/(mode+"_history.json"),history);torch.save({"risk":model.state_dict(),"mode":mode},args.out/(mode+"_risk.pt"))
        with torch.no_grad():
            hp=predict(model,hx,mode=="reset");eligible=hv.bool();eligible[:,0]=False
            levels=torch.quantile(hp[eligible],hp.new_tensor([.25,.5,.75])).cpu().tolist()
            thresholds[mode]={str(qtile):value for qtile,value in zip((25,50,75),levels)}
            diagnostics[mode]={"fit_mae":float(((predict(model,x,mode=="reset")-y).abs()*valid).sum()/valid.sum()),
                               "calibration_mae":float(((hp-hy).abs()*hv).sum()/hv.sum()),"calibration_target_mean":float((hy*hv).sum()/hv.sum()),
                               "calibration_prediction_mean":float((hp*hv).sum()/hv.sum()),"parameters":sum(p.numel() for p in model.parameters())}
    q.write(args.out/"thresholds.json",thresholds);q.write(args.out/"risk_diagnostics.json",diagnostics)
    print("thresholds",thresholds,"diagnostics",diagnostics,flush=True)
    result={"promoted":False};parity=[]
    for length,samples in dev.items():
        baseline,dense=evaluate(exact,samples);result[length]={"baseline":baseline};torch.save(dense,args.out/(length+"_dense_predictions.pt"))
        wrappers={"output_k2":RGBFlowStream(exact),"output_k4":K4OutputFlow(exact)}
        for name,model in models.items():
            for percentile,threshold in thresholds[name].items():wrappers[name+"_q"+percentile]=ScheduledFlow(exact,model,threshold,reset_h=name=="reset")
        wrappers["carry_q50_h_reset"]=ScheduledFlow(exact,models["carry"],thresholds["carry"]["50"],reset_h=True)
        for name,wrapper in wrappers.items():
            measured,preds=evaluate(wrapper,samples)
            measured["quality_gate"]=quality_gate(baseline["metrics"],measured["metrics"])
            measured["speedup"]=baseline["mean_ms"]/measured["mean_ms"]
            measured["screen_pass"]=measured["quality_gate"]["pass"] and measured["speedup"]>=1/.95
            if name!="output_k2":measured["speedup_vs_k2"]=result[length]["output_k2"]["mean_ms"]/measured["mean_ms"]
            count=0
            for ci,tele in enumerate(measured["telemetry"]):
                for t,info in enumerate(tele["frames"]):
                    if info["full_refresh"]:assert torch.equal(preds[ci][t],dense[ci][t]);count+=1
            parity.append({"length":length,"policy":name,"exact_refresh_comparisons":count})
            result[length][name]=measured;torch.save(preds,args.out/(length+"_"+name+"_predictions.pt"))
            q.write(args.out/"results.json",result);q.write(args.out/"parity.json",parity)
            print(length,name,"raw",measured["metrics"]["balanced"]["absrel_raw"],"F1",measured["metrics"]["balanced"]["boundary_f1"],
                  "ms",measured["mean_ms"],"skip",measured["skip_fraction"],"gate",measured["quality_gate"],flush=True)
    for r in protocol["code"]+protocol["development"]+[protocol["checkpoint"],cache_rec,protocol["parent_protocol"]]:assert file_record(r["path"])==r
    q.write(args.out/"integrity.json",{"input_hashes_unchanged":True,"exact_refresh_comparisons":sum(r["exact_refresh_comparisons"] for r in parity)})


if __name__=="__main__":main()
