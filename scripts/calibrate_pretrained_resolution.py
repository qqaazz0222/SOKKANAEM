"""Resolution adaptation of Q0 calibration ONLY; frozen pretrained shape.

Not encoder-decoder joint training, not an encoder-only causal comparison.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.architecture_readout_study import strict_gate
from scripts.pretrained_quality_study import summarize
from sokkanaem.model import from_checkpoint
from sokkanaem.sharpness import norm_field
from sokkanaem.protocol import file_record


@torch.no_grad()
def cache(model,samples):
    result=[]
    for i,s in enumerate(samples):
        disp,feat=model._shape(s["rgb"].cuda().float())
        result.append({"dn":norm_field(disp,torch.ones_like(disp)).cpu(),"feat":feat.cpu(),
                       "gt":s["gt"],"valid":s["valid"],"source":s["source"],"scene":s["scene"],"pairs":s["pairs"]})
        if (i+1)%32==0:print("calibration_cache",i+1,"/",len(samples),flush=True)
    return result


def predict(calib,dn,feat):
    scale,shift=calib(feat).unbind(-1)
    disparity=F.softplus(scale)[:,None,None,None]*dn+shift[:,None,None,None]
    return (1/disparity.clamp_min(1/150.)).clamp(.3,150.)


@torch.no_grad()
def evaluate(calib,data):
    preds=[predict(calib,s["dn"].cuda(),s["feat"].cuda()).cpu() for s in data]
    return summarize(preds,data),preds


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--parent",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True);ap.add_argument("--steps",type=int,default=1600)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("Use a new output folder")
    p=json.loads((args.parent/"protocol.json").read_text())
    assert file_record(p["q_checkpoint"]["path"])==p["q_checkpoint"]
    args.out.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(735);torch.backends.cudnn.benchmark=False
    model=from_checkpoint(p["q_checkpoint"]["path"],"cuda").eval().requires_grad_(False)
    state=torch.load(p["q_checkpoint"]["path"],weights_only=False,map_location="cpu")
    model.load_state_dict(state.get("ema") or state.get("model") or state,strict=True)
    model.infer_size=224
    train=[s for s in torch.load(p["train_data"]["path"],weights_only=False) if s["source"] in ("tum","bonn")]
    dev=torch.load(p["dev_data"]["path"],weights_only=False)
    tf={str(Path(x).resolve()) for s in train for pair in s["pairs"] for x in pair}
    df={str(Path(x).resolve()) for s in dev for pair in s["pairs"] for x in pair}
    assert not tf&df
    train_cache=cache(model,train);dev_cache=cache(model,dev)
    torch.save(train_cache,args.out/"train_cache.pt");torch.save(dev_cache,args.out/"dev_cache.pt")
    initial,_=evaluate(model.calib,dev_cache)
    calib=model.calib;calib.requires_grad_(True)
    # The optimizer owns only calibration, and the frozen shape network is freed.
    del model
    protocol={"role":"adaptive_development_resolution_calibration_not_final","parent":file_record(args.parent/"protocol.json"),
              "checkpoint":p["q_checkpoint"],"infer_size":224,"steps":args.steps,"seed":735,"lr":.0001,"batch":8,
              "train_frames":sum(len(s["gt"]) for s in train),"development_frames":sum(len(s["gt"]) for s in dev),
              "objective":"valid-pixel log-depth L1 only; frozen shape predictions and pooled features",
              "train_cache":file_record(args.out/"train_cache.pt"),"dev_cache":file_record(args.out/"dev_cache.pt"),
              "trainable_parameters":sum(x.numel() for x in calib.parameters()),
              "code":[file_record(x) for x in [__file__,"scripts/pretrained_quality_study.py","scripts/architecture_readout_study.py",
                         "sokkanaem/qmodel.py","sokkanaem/sharpness.py"]]}
    q.write(args.out/"protocol.json",protocol)
    data={k:torch.cat([s[k] for s in train_cache]).cuda() for k in ("dn","feat","gt","valid")}
    opt=torch.optim.AdamW(calib.parameters(),lr=.0001,weight_decay=.0001)
    gen=torch.Generator(device="cuda").manual_seed(735);history=[]
    for step in range(1,args.steps+1):
        ix=torch.randint(len(data["gt"]),(8,),device="cuda",generator=gen)
        x={k:v[ix] for k,v in data.items()}
        pred=predict(calib,x["dn"],x["feat"])
        valid=x["valid"].bool() & torch.isfinite(x["gt"]) & (x["gt"]>0) & (x["gt"]<150)
        loss=(pred[valid].log()-x["gt"][valid].log()).abs().mean()
        if not torch.isfinite(loss):raise ValueError("nonfinite calibration loss")
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(calib.parameters(),1.);opt.step()
        if step==1 or step%200==0:
            row={"step":step,"loss":float(loss.detach())};history.append(row);print("calibration",row,flush=True)
    calib.eval();final,preds=evaluate(calib,dev_cache)
    reference=json.loads((args.parent/"pretrained_comparison.json").read_text())["points"]
    result={"initial":initial,"final":final,"history":history,"promoted":False,
            "gate_vs_native224":strict_gate(reference["native224"]["metrics"],final["metrics"]),
            "gate_vs_initial_q0_224":strict_gate(initial["metrics"],final["metrics"]),
            "saturated_pixel_fraction":float(torch.cat([(p>=150).flatten() for p in preds]).float().mean())}
    torch.save({"calib":calib.state_dict(),"infer_size":224,"q0_sha256":p["q_checkpoint"]["sha256"],
                "steps":args.steps},args.out/"calibration.pt")
    torch.save(preds,args.out/"predictions.pt")
    q.write(args.out/"results.json",result)
    assert file_record(p["q_checkpoint"]["path"])==p["q_checkpoint"]
    print("FINAL_CALIBRATION",final["metrics"]["balanced"],result["gate_vs_native224"],flush=True)


if __name__=="__main__":main()
