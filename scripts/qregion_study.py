"""Matched local/global bounded output repair, teacher-retention objective."""
import argparse
import copy
import json
from pathlib import Path
import sys
import cv2
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact
from scripts.qstream_study import evaluate,quality_gate
from sokkanaem.protocol import file_record,FINAL_TEST_SEQUENCES,runtime_versions
from sokkanaem.qflow import RGBFlowStream,warp_depth
from sokkanaem.qquality import depth_retention
from sokkanaem.qregion_repair import RegionRepair,RegionFlowStream,no_harm_loss


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--steps",type=int,default=1200);args=ap.parse_args()
    if args.out.exists():raise FileExistsError("New output required")
    if args.steps<1:raise ValueError("Positive steps required")
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    parent=Path("work_dirs/qrecurrent_20260909")
    pp=json.loads((parent/"protocol.json").read_text());cache_rec=json.loads((parent/"cache_record.json").read_text())
    assert file_record(cache_rec["path"])==cache_rec
    exact=load_exact(pp["checkpoint"])
    train=torch.load(cache_rec["path"],weights_only=False)
    paths=[Path("work_dirs/qstream_20260908/development_data.pt"),Path("work_dirs/qquality_audit_20260908/long_development_data.pt")]
    dev={n:torch.load(p,weights_only=False) for n,p in zip(("L32","L256"),paths)}
    tf={str(Path(p).resolve()) for s in train for pair in s["pairs"] for p in pair}
    df={str(Path(p).resolve()) for samples in dev.values() for s in samples for pair in s["pairs"] for p in pair}
    assert not tf&df and not any(set(Path(p).parts)&set(FINAL_TEST_SEQUENCES) for p in tf|df)
    models={}
    for mode in ("local","global"):
        torch.manual_seed(740);models[mode]=RegionRepair(local=mode=="local").cuda()
    assert all(torch.equal(a,b) for a,b in zip(models["local"].parameters(),models["global"].parameters()))
    args.out.mkdir(parents=True)
    protocol={"role":"K2_output_region_training_reused_development_no_final", "checkpoint":pp["checkpoint"],
              "training_cache":cache_rec,"parent_protocol":file_record(parent/"protocol.json"),"development":[file_record(p) for p in paths],
              "training_clips":len(train),"training_frames":sum(len(s["rgb"]) for s in train),"steps_each":args.steps,
              "seed":740,"lr":.0005,"batch":4,"parameters_each":sum(p.numel() for p in models["local"].parameters()),
              "comparison":"same architecture/initial weights/order/loss/budget; trainlocal/global x applylocal/global; mask from RGB correspondence only",
              "editable":"complement of same DIS reliability incl bounds/FB/photo/erosion, not true disocclusion; 5x5 support taper, +/- .03logdepth; exact copy outside support",
              "objective":"5Q0logdepth +10Q0multiscalegradient +10positive_increase_in_Q0depth_error +10positive_increase_in_Q0gradient_error; NO GTflat penalty",
              "limits":"no-harm is a training penalty NOT guarantee; denseCNN computes entire frame; no sparse-compute or SSM novelty claim; different loss from previous feature studies, not a loss-only ablation",
              "gate":"unchanged quality_gate and >=5pct mean latency reduction", "timing":"synchronized GPUresident includes CPUflow/transfers/CNN/telemetry; excludes initialIO/H2D; RTX4090 not edge",
              "runtime":runtime_versions(),"opencv":cv2.__version__,
              "code":[file_record(p) for p in [__file__,"sokkanaem/qregion_repair.py","sokkanaem/qflow.py","sokkanaem/qquality.py",
                        "sokkanaem/qstream.py","sokkanaem/qmodel.py","scripts/qquality_study.py","scripts/qstream_study.py","scripts/quality_refinement.py","sokkanaem/sharpness.py"]]}
    q.write(args.out/"protocol.json",protocol)
    cache=[]
    with torch.no_grad():
        for s in train:
            for t in (1,3):
                previous=torch.cat((s["teacher"][t-1:t],s["rgb"][t-1:t]),1).cuda()
                moved,reliable=warp_depth(previous,s["flows"][t-1],s["reliable"][t-1],"nearest")
                cache.append({"current":s["rgb"][t:t+1],"warped_rgb":moved[:,1:].cpu(),"coarse":moved[:,:1].cpu(),
                              "editable":(~reliable).cpu(),"teacher":s["teacher"][t:t+1],"pairs":s["pairs"][t],"source":s["source"]})
    assert len(cache)==384;del train
    torch.save(cache,args.out/"training_cache.pt");q.write(args.out/"cache_record.json",file_record(args.out/"training_cache.pt"))
    order=torch.randint(len(cache),(args.steps,4),generator=torch.Generator().manual_seed(740)).tolist()
    q.write(args.out/"training_order.json",order)
    for name,model in models.items():
        opt=torch.optim.AdamW(model.parameters(),lr=.0005);history=[]
        for step,indices in enumerate(order,1):
            b={k:torch.cat([cache[i][k] for i in indices]).cuda() for k in ("current","warped_rgb","coarse","editable","teacher")}
            pred=model(b["current"],b["warped_rgb"],b["coarse"],b["editable"]).clamp(exact.q.d_min,exact.q.d_max)
            depth,edge=depth_retention(pred,b["teacher"]);dh,eh=no_harm_loss(pred,b["coarse"],b["teacher"])
            loss=5*depth+10*edge+10*dh+10*eh
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step==1 or step%200==0 or step==args.steps:
                row={"step":step,"loss":float(loss.detach()),"depth":float(depth.detach()),"edge":float(edge.detach()),"depth_harm":float(dh.detach()),"edge_harm":float(eh.detach())}
                history.append(row);q.write(args.out/(name+"_history.json"),history);print(name,row,flush=True)
        model.eval().requires_grad_(False);torch.save({"repair":model.state_dict(),"local":model.local},args.out/(name+"_repair.pt"))
    result={"promoted":False};parity=[]
    for length,samples in dev.items():
        baseline,dense=evaluate(exact,samples);result[length]={"baseline":baseline}
        torch.save(dense,args.out/(length+"_dense_predictions.pt"))
        base,bp=evaluate(RGBFlowStream(exact),samples);base["quality_gate"]=quality_gate(baseline["metrics"],base["metrics"])
        base["speedup"]=baseline["mean_ms"]/base["mean_ms"];base["screen_pass"]=base["quality_gate"]["pass"] and base["speedup"]>=1/.95
        result[length]["output_flow"]=base;torch.save(bp,args.out/(length+"_output_flow_predictions.pt"))
        for trained,model in models.items():
            for local in (True,False):
                repair=copy.deepcopy(model);repair.local=local;name=trained+("_local" if local else "_global")
                measured,preds=evaluate(RegionFlowStream(exact,repair),samples)
                measured["quality_gate"]=quality_gate(baseline["metrics"],measured["metrics"])
                measured["speedup"]=baseline["mean_ms"]/measured["mean_ms"]
                measured["screen_pass"]=measured["quality_gate"]["pass"] and measured["speedup"]>=1/.95
                full=0
                for ci,tele in enumerate(measured["telemetry"]):
                    for t,info in enumerate(tele["frames"]):
                        if info["full_refresh"]:assert torch.equal(preds[ci][t],dense[ci][t]);full+=1
                        elif local:assert info["outside_edit_max_abs"]==0.
                parity.append({"length":length,"model":name,"exact_refresh_comparisons":full,"outside_support_exact":local})
                result[length][name]=measured;torch.save(preds,args.out/(length+"_"+name+"_predictions.pt"))
                q.write(args.out/"results.json",result);q.write(args.out/"parity.json",parity)
                print(length,name,measured["metrics"]["balanced"],"ms",measured["mean_ms"],measured["quality_gate"],flush=True)
    for r in protocol["code"]+protocol["development"]+[protocol["checkpoint"],cache_rec,protocol["parent_protocol"]]:assert file_record(r["path"])==r
    q.write(args.out/"integrity.json",{"input_hashes_unchanged":True,"full_refresh_comparisons":sum(p["exact_refresh_comparisons"] for p in parity),"local_outside_edit_error_zero":True})


if __name__=="__main__":main()
