"""Stage 1 error decomposition and stage 2 resolution-matched pretrained anchor.

Q0 is a complete pretrained system comparison, not an encoder-only ablation.
No network access needed when HF_HUB_OFFLINE=1; no paper test or promotion.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
import time
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.architecture_readout_study import strict_gate
from sokkanaem.model import from_checkpoint
from sokkanaem.joint_detail_decoder import JointDetailDecoder
from sokkanaem.quality_diagnostics import region_diagnostics,clip_log_scale_correction
from sokkanaem.protocol import file_record,check_final_test,runtime_versions


def aggregate_regions(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[(row["source"],row["scene"])].append(row["diagnostics"])
    source=defaultdict(list)
    missing=defaultdict(int)
    for (src,scene),items in grouped.items():
        means={}
        for key in items[0]:
            vals=[r[key] for r in items if r[key] is not None]
            missing[key]+=len(items)-len(vals)
            means[key]=sum(vals)/len(vals) if vals else None
        source[src].append(means)
    per_source={}
    for src,items in source.items():
        per_source[src]={k:statistics.mean(v) if (v:=[r[k] for r in items if r[k] is not None]) else None for k in items[0]}
    balanced={k:statistics.mean(v) if (v:=[r[k] for r in per_source.values() if r[k] is not None]) else None for k in next(iter(per_source.values()))}
    return {"balanced":balanced,"sources":per_source,"missing_clip_counts":dict(missing),"clips":rows}


def summarize(preds,samples):
    metric_rows=[];regions=[]
    for p,s in zip(preds,samples):
        p=p.cuda()
        metric_rows.append({"source":s["source"],"scene":s["scene"],"metrics":q.score(p,s)})
        regions.append({"source":s["source"],"scene":s["scene"],
                        "diagnostics":region_diagnostics(p,s["gt"].cuda(),s["valid"].cuda())})
    return {"metrics":q.aggregate(metric_rows),"regions":aggregate_regions(regions)}


@torch.no_grad()
def decoder_predictions(model,samples):
    return [model(s["tokens"].cuda().unbind(1),s["rgb"].cuda().float()).cpu() for s in samples]


@torch.no_grad()
def stage_one(base,parent,out):
    data=torch.load(parent/"dev_tokens.pt",weights_only=False)
    train=[s for s in torch.load(parent/"train_tokens.pt",weights_only=False) if s["source"] in ("tum","bonn")]
    results={}
    for name in ("baseline","joint_detail"):
        model=JointDetailDecoder(base.decoder,name=="joint_detail").cuda().eval()
        if name=="joint_detail":
            model.load_state_dict(torch.load(parent/"joint_detail.pt",weights_only=False)["state_dict"])
        corrections=defaultdict(list)
        for s in train:
            pred=model(s["tokens"].cuda().unbind(1),s["rgb"].cuda().float())
            corrections[s["source"]].append(clip_log_scale_correction(pred,s["gt"].cuda(),s["valid"].cuda()))
        source_means={src:statistics.mean(v) for src,v in corrections.items()}
        factor=math.exp(statistics.mean(source_means.values()))
        preds=decoder_predictions(model,data)
        torch.save(preds,out/(name+"_predictions.pt"))
        results[name]={"raw":summarize(preds,data),"training_scale_factor":factor,
                       "training_source_log_factors":source_means,
                       "train_calibrated":summarize([p*factor for p in preds],data)}
        print("ERROR_DECOMPOSITION",name,"training_scale",factor,"raw",results[name]["raw"]["metrics"]["balanced"],
              "train_scaled",results[name]["train_calibrated"]["metrics"]["balanced"],flush=True)
    results["note"]="Global scale fitted on training TUM/Bonn only, equal clip/source weighting in log space. Clip-median scores are GT-assisted diagnostics and never deployed. Boundary recall tolerance is NOT small-object instance recall."
    q.write(out/"error_decomposition.json",results)
    return data


@torch.no_grad()
def predict_point(model,data,kind,size):
    if kind=="q":model.infer_size=size
    # Warm up the same GPU-resident input path, outside the measured set.
    def one(frame,state):
        if kind=="native":
            x=F.interpolate(frame,size=(size,size),mode="bilinear",align_corners=False)
            pred,state,_=model.step(x,state)
            return F.interpolate(pred,size=(256,256),mode="bilinear",align_corners=False),state
        pred,state,_=model.step(frame,state)
        return pred,state
    frame=data[0]["rgb"][0:1].cuda().float();state=None
    for _ in range(20):_,state=one(frame,state)
    torch.cuda.synchronize()
    preds=[];milliseconds=[]
    for s in data:
        state=None;frames=[]
        for rgb in s["rgb"]:
            frame=rgb[None].cuda().float()
            torch.cuda.synchronize();start=time.perf_counter()
            pred,state=one(frame,state)
            torch.cuda.synchronize();milliseconds.append((time.perf_counter()-start)*1000)
            frames.append(pred.cpu())
        preds.append(torch.cat(frames))
    measured=summarize(preds,data)
    measured["latency"]={"median_ms":statistics.median(milliseconds),
                         "p95_ms":float(torch.tensor(milliseconds).quantile(.95)),
                         "frames":len(milliseconds),"per_frame_ms":milliseconds,
                         "scope":"RTX4090 FP32, GPU-resident input, synchronized Python call incl resize/normalization/output conversion; excludes IO/H2D, not Nano or throughput"}
    measured["parameters"]=sum(p.numel() for p in model.parameters())
    return measured,preds


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--parent",type=Path,default=Path("work_dirs/joint_decoder_20260908_v2"));args=ap.parse_args()
    if args.out.exists():raise FileExistsError("Immutable study directory exists")
    p=json.loads((args.parent/"protocol.json").read_text())
    original=json.loads(Path(p["parent"]["path"]).read_text())
    assert check_final_test(original["manifest"]["path"])=="development_validation"
    assert file_record(p["checkpoint"]["path"])==p["checkpoint"]
    args.out.mkdir(parents=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    qpath=Path("work_dirs/q0-calib-s0/latest.pt")
    protocol={"role":"adaptive_error_analysis_and_pretrained_system_comparison_not_final",
              "checkpoint":p["checkpoint"],"q_checkpoint":file_record(qpath),
              "q_config":file_record(qpath.with_name("config.toml")),
              "dev_data":file_record(args.parent/"dev_tokens.pt"),"train_data":file_record(args.parent/"train_tokens.pt"),
              "joint_checkpoint":file_record(args.parent/"joint_detail.pt"),"runtime":runtime_versions(),
              "gpu":torch.cuda.get_device_name(),"common_resolution":224,"score_resolution":256,
              "matched_comparison":"native224 vs Q0_224; both resize same cached RGB256 to224. Both checkpoints change inference resolution without retraining.",
              "reference_only":"native256 and Q0_518 are different compute budgets. Q0 includes a different decoder, calibration and pretraining; not encoder-only causal evidence.",
              "code":[file_record(x) for x in [__file__,"sokkanaem/quality_diagnostics.py","sokkanaem/joint_detail_decoder.py",
                         "scripts/quality_refinement.py","sokkanaem/model.py","sokkanaem/qmodel.py","sokkanaem/sharpness.py"]]}
    q.write(args.out/"protocol.json",protocol)
    base=from_checkpoint(p["checkpoint"]["path"],"cuda").eval().requires_grad_(False)
    data=stage_one(base,args.parent,args.out)
    comparisons={}
    for size in (224,256):
        measured,preds=predict_point(base,data,"native",size)
        comparisons[f"native{size}"]=measured;torch.save(preds,args.out/f"native{size}_predictions.pt")
        print("PRETRAINED_REFERENCE",f"native{size}",measured["metrics"]["balanced"],"ms",measured["latency"]["median_ms"],flush=True)
    del base
    model=from_checkpoint(qpath,"cuda").eval().requires_grad_(False)
    state=torch.load(qpath,weights_only=False,map_location="cpu");state=state.get("ema") or state.get("model") or state
    model.load_state_dict(state,strict=True)
    for size in (224,518):
        measured,preds=predict_point(model,data,"q",size)
        comparisons[f"q0_{size}"]=measured;torch.save(preds,args.out/f"q0_{size}_predictions.pt")
        print("PRETRAINED_REFERENCE",f"q0_{size}",measured["metrics"]["balanced"],"ms",measured["latency"]["median_ms"],flush=True)
    result={"points":comparisons,"matched_gate":strict_gate(comparisons["native224"]["metrics"],comparisons["q0_224"]["metrics"]),
            "promoted":False,"encoder_only_ablation_completed":False}
    q.write(args.out/"pretrained_comparison.json",result)
    assert file_record(p["checkpoint"]["path"])==p["checkpoint"]
    assert file_record(qpath)==protocol["q_checkpoint"]


if __name__=="__main__":main()
