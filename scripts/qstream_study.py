"""Q0-equivalent streaming and actual full-inference skipping on development L32.

Stage 1/2 only. Output hold is explicitly a control, not the final SSM model.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from sokkanaem.data import load_manifest
from sokkanaem.model import from_checkpoint
from sokkanaem.protocol import file_record,check_final_test,runtime_versions
from sokkanaem.qstream import Q0ExactStream,ChangeAwareQ0


def quality_gate(baseline,candidate):
    a,b=baseline["balanced"],candidate["balanced"]
    checks={"raw_absrel_within_1pct":b["absrel_raw"]<=a["absrel_raw"]*1.01,
            "edge_absrel_within_1pct":b["absrel_edge"]<=a["absrel_edge"]*1.01,
            "boundary_f1_drop_at_most_0.005":b["boundary_f1"]>=a["boundary_f1"]-.005,
            "overshoot_within_1pct":b["overshoot"]<=a["overshoot"]*1.01,
            "flat_tv_within_1pct":b["flat_tv"]<=a["flat_tv"]*1.01,
            "per_source_raw_within_2pct":all(candidate["sources"][s]["absrel_raw"]<=v["absrel_raw"]*1.02 for s,v in baseline["sources"].items())}
    return {"pass":all(checks.values()),"checks":checks,
            "scope":"Predeclared Q0-relative development screen, not former native-improvement gate or a final guarantee."}


@torch.no_grad()
def evaluate(model,samples):
    rows=[];predictions=[];latency=[];telemetry=[]
    frame=samples[0]["rgb"][0:1].cuda()
    state=None
    for _ in range(16):_,state,_=model.step(frame,state)
    torch.cuda.synchronize()
    for ci,s in enumerate(samples):
        preds=[];state=None;counts=[]
        for rgb in s["rgb"]:
            frame=rgb[None].cuda()
            torch.cuda.synchronize();start=time.perf_counter()
            pred,state,info=model.step(frame,state)
            torch.cuda.synchronize();latency.append((time.perf_counter()-start)*1000)
            preds.append(pred.cpu());counts.append(info)
        pred=torch.cat(preds);predictions.append(pred)
        rows.append({"source":s["source"],"scene":s["scene"],"metrics":q.score(pred.cuda(),s)})
        telemetry.append({"clip":ci,"frames":counts})
        print("evaluated",ci+1,"/",len(samples),flush=True)
    frames=sum(len(t["frames"]) for t in telemetry)
    calls=sum(x["backbone_calls"] for t in telemetry for x in t["frames"])
    return {"metrics":q.aggregate(rows),"full_backbone_calls":calls,"frames":frames,
            "skip_fraction":1-calls/frames,"median_ms":statistics.median(latency),
            "mean_ms":statistics.mean(latency),"p95_ms":float(np.quantile(latency,.95)),
            "per_frame_ms":latency,"telemetry":telemetry},predictions


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise FileExistsError("Use a new study directory")
    manifest=Path("manifests/acc_real_L32.json")
    assert check_final_test(manifest)=="development_validation"
    selected=[]
    for src,ds in load_manifest(manifest):
        groups=defaultdict(list)
        for i in range(len(ds)):groups[q.scene(q.paths(ds,i)[0])].append(i)
        for scene,indices in groups.items():
            for j in np.linspace(0,len(indices)-1,2).astype(int):selected.append((src,ds,indices[j]))
    assert len(selected)==8
    args.out.mkdir(parents=True);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    checkpoint=Path("work_dirs/q0-calib-s0/latest.pt")
    policies={"conservative":{"tau_on":1e-4,"tau_off":5e-5,"refresh_every":8,"max_active":0.},
              "moderate":{"tau_on":1e-3,"tau_off":5e-4,"refresh_every":8,"max_active":.01},
              "periodic_hold4":{"refresh_every":4,"periodic_only":True}}
    protocol={"role":"Q0_relative_streaming_development_not_final","checkpoint":file_record(checkpoint),
              "config":file_record(checkpoint.with_name("config.toml")),"manifest":file_record(manifest),
              "policies":policies,"inference_resolution":518,"score_resolution":256,
              "selection":[{"source":s,"pairs":q.paths(d,i)} for s,d,i in selected],
              "quality_budget":{"raw_absrel_relative":.01,"edge_absrel_relative":.01,"f1_absolute_drop":.005,
                                "overshoot_relative":.01,"flat_tv_relative":.01,"per_source_raw_relative":.02},
              "speed_screen":"quality pass AND mean GPU-resident latency at least 5% below dense Q0",
              "timing":"FP32 RTX4090 resident input, synchronized Python invocation, incl detector/preprocess/model, excludes file IO/H2D; no device deployment claim",
              "runtime":runtime_versions(),"gpu":torch.cuda.get_device_name(),
              "code":[file_record(x) for x in [__file__,"sokkanaem/qstream.py","sokkanaem/qmodel.py",
                         "sokkanaem/detector.py","sokkanaem/sharpness.py","scripts/quality_refinement.py"]]}
    q.write(args.out/"protocol.json",protocol)
    samples=[]
    for src,ds,i in selected:
        rgb,gt,valid=ds[i]
        samples.append({"rgb":rgb,"gt":gt,"valid":valid.bool(),"source":src,
                        "scene":q.scene(q.paths(ds,i)[0]),"pairs":q.paths(ds,i)})
    torch.save(samples,args.out/"development_data.pt")
    model=from_checkpoint(checkpoint,"cuda").eval().requires_grad_(False)
    weights=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model.load_state_dict(weights.get("ema") or weights.get("model") or weights,strict=True)
    exact=Q0ExactStream(model)
    parity=[]
    with torch.no_grad():
        for size in (224,518):
            model.infer_size=size
            for ci,s in enumerate(samples):
                for t in (0,15,31):
                    frame=s["rgb"][t:t+1].cuda()
                    original=model(frame);before=exact.backbone_calls
                    new,_,_=exact.step(frame)
                    rel=float(((new-original).abs()/original).mean());maxerr=float((new-original).abs().max())
                    assert exact.backbone_calls-before==1
                    torch.testing.assert_close(new,original,rtol=1e-5,atol=1e-5)
                    parity.append({"size":size,"clip":ci,"frame":t,"relative_mean":rel,"max_abs":maxerr})
    q.write(args.out/"parity.json",{"comparisons":parity,"pass":True,"backbone_calls_per_frame":1})
    print("PARITY PASS",len(parity),"comparisons",flush=True)
    model.infer_size=518
    baseline,preds=evaluate(exact,samples);torch.save(preds,args.out/"dense_predictions.pt")
    results={"baseline":baseline,"policies":{},"promoted":False,"selective_ssm_implemented":False}
    print("BASELINE",baseline["metrics"]["balanced"],"mean_ms",baseline["mean_ms"],flush=True)
    for name,config in policies.items():
        wrapper=ChangeAwareQ0(exact,**config)
        measured,preds=evaluate(wrapper,samples)
        measured["quality_gate"]=quality_gate(baseline["metrics"],measured["metrics"])
        measured["mean_latency_speedup"]=baseline["mean_ms"]/measured["mean_ms"]
        measured["screen_pass"]=measured["quality_gate"]["pass"] and measured["mean_ms"]<=.95*baseline["mean_ms"]
        results["policies"][name]=measured
        torch.save(preds,args.out/(name+"_predictions.pt"))
        q.write(args.out/"results.json",results)
        print(name,measured["metrics"]["balanced"],"skip",measured["skip_fraction"],"mean_ms",measured["mean_ms"],
              "gate",measured["quality_gate"],flush=True)
    assert file_record(checkpoint)==protocol["checkpoint"]
    q.write(args.out/"results.json",results)


if __name__=="__main__":main()
