"""Teacher-feature trained selective SSM, actual Q0 encoder skips on L32.

Q0 weights stay frozen. Only a first feature-retention screen, not deployment.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qstream_study import evaluate,quality_gate
from sokkanaem.qstream import Q0ExactStream
from sokkanaem.qdelta_ssm import QDeltaSSM
from sokkanaem.detector import ChangeDetector
from sokkanaem.model import from_checkpoint
from sokkanaem.protocol import file_record


class HybridStream(torch.nn.Module):
    def __init__(self,exact,adapter,adaptive=True):
        super().__init__();self.exact=exact;self.adapter=adapter;self.adaptive=adaptive
        self.detector=ChangeDetector(patch_size=14,tau_on=1e-3,tau_off=5e-4,keyframe_every=4,dilate=True)
    @torch.no_grad()
    def step(self,frame,state=None):
        image=F.interpolate(frame,size=(518,518),mode="bilinear",align_corners=False)
        force=state is None or state["age"]>=3
        score=None if force else F.avg_pool2d((image-state["anchor"]).square().mean(1,keepdim=True),14)[:,0]
        mask,det=self.detector.gate(score,1,37,37,frame.device,None if state is None else state["det"])
        active=float(mask.mean());refresh=force or (self.adaptive and active>.4)
        if refresh:
            encoded=self.exact.encode(frame);features=self.adapter.refresh(encoded)
            state={"features":features,"anchor":image.clone(),"age":0,"det":det};updated=0
        else:
            features,info=self.adapter.update(frame,mask,state["features"])
            state={**state,"features":features,"age":state["age"]+1,"det":det};updated=info["updated_patches"]
        pred=self.exact.decode(state["features"])
        return pred,state,{"full_refresh":refresh,"backbone_calls":int(refresh),"active_ratio":active,
                           "updated_patches":updated,"cache_age":state["age"],"mode":"selective_ssm_feature_prediction"}


@torch.no_grad()
def teacher_cache(exact,samples):
    result=[]
    for i,s in enumerate(samples):
        encoded=exact.encode(s["rgb"].cuda().float())
        result.append({"rgb":s["rgb"],"features":[x.cpu().half() for x in encoded["features"]],
                       "pooled":encoded["pooled"].cpu(),"grid":encoded["grid"],"size":encoded["size"],
                       "source":s["source"],"pairs":s["pairs"]})
        if (i+1)%16==0:print("teacher cache",i+1,"/",len(samples),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--parent",type=Path,default=Path("work_dirs/qstream_20260908"))
    ap.add_argument("--steps",type=int,default=600);args=ap.parse_args()
    if args.out.exists():raise FileExistsError("Use a new output directory")
    pp=json.loads((args.parent/"protocol.json").read_text())
    assert file_record(pp["checkpoint"]["path"])==pp["checkpoint"]
    args.out.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(736)
    torch.backends.cudnn.benchmark=False
    model=from_checkpoint(pp["checkpoint"]["path"],"cuda").eval().requires_grad_(False)
    original=torch.load(pp["checkpoint"]["path"],map_location="cpu",weights_only=False)
    model.load_state_dict(original.get("ema") or original.get("model") or original,strict=True)
    exact=Q0ExactStream(model)
    source=Path("work_dirs/architecture_readout_20260908/train_cache.pt")
    all_train=torch.load(source,weights_only=False);train=[]
    for name in ("tum","bonn","vkitti2"):train.extend([s for s in all_train if s["source"]==name][:16])
    dev=torch.load(args.parent/"development_data.pt",weights_only=False)
    tf={str(Path(p).resolve()) for s in train for pair in s["pairs"] for p in pair}
    df={str(Path(p).resolve()) for s in dev for pair in s["pairs"] for p in pair}
    assert not tf&df
    protocol={"role":"adaptive_Q0_feature_prediction_development_not_final","checkpoint":pp["checkpoint"],
              "parent":file_record(args.parent/"protocol.json"),"train_source":file_record(source),
              "train_selection":[{"source":s["source"],"pairs":s["pairs"]} for s in train],
              "steps":args.steps,"seed":736,"lr":.0003,"train_clips":48,"train_frames":192,
              "objective":"mean L1 to frozen Q0 stage features + 0.1 raw pooled-feature L1; no GT losses and no decoded-depth loss",
              "policy":"K4; anchor-relative MSE tau_on=.001/tau_off=.0005; adaptive full refresh when active > .4",
              "training_policy":"one teacher refresh at t0, three selective predicted updates; no motion fallback during training",
              "limits":"CLS and calibration pooled features get learned global deltas; only inactive patch tokens/SSM states are copied, not the whole output",
              "code":[file_record(x) for x in [__file__,"sokkanaem/qdelta_ssm.py","sokkanaem/qstream.py",
                         "sokkanaem/ssm.py","scripts/qstream_study.py","sokkanaem/detector.py"]]}
    q.write(args.out/"protocol.json",protocol)
    cache=teacher_cache(exact,train);torch.save(cache,args.out/"teacher_cache.pt")
    adapter=QDeltaSSM().cuda();opt=torch.optim.AdamW(adapter.parameters(),lr=.0003,weight_decay=.0001)
    detector=ChangeDetector(patch_size=14,tau_on=.001,tau_off=.0005,keyframe_every=4,dilate=True)
    gen=torch.Generator().manual_seed(736);history=[]
    for step in range(1,args.steps+1):
        s=cache[int(torch.randint(len(cache),(),generator=gen))]
        rgb=s["rgb"].cuda().float();fs=[x.cuda().float() for x in s["features"]];pooled=s["pooled"].cuda()
        initial={"features":[x[:1] for x in fs],"pooled":pooled[:1],"grid":s["grid"],"size":s["size"]}
        state=adapter.refresh(initial)
        images=F.interpolate(rgb,size=(518,518),mode="bilinear",align_corners=False)
        _,det=detector.gate(None,1,37,37,rgb.device,None)
        losses=[]
        for t in range(1,len(rgb)):
            score=F.avg_pool2d((images[t:t+1]-images[:1]).square().mean(1,keepdim=True),14)[:,0]
            mask,det=detector.gate(score,1,37,37,rgb.device,det)
            state,_=adapter.update(rgb[t:t+1],mask,state)
            feature_loss=sum((a-b[t:t+1]).abs().mean() for a,b in zip(state["features"],fs))/len(fs)
            pooled_loss=(state["pooled"]-pooled[t:t+1]).abs().mean()
            losses.append(feature_loss+.1*pooled_loss)
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise ValueError("nonfinite adapter loss")
        opt.zero_grad(set_to_none=True)
        if loss.requires_grad:
            loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);opt.step()
        if step==1 or step%100==0:
            row={"step":step,"loss":float(loss.detach())};history.append(row);print("delta_ssm",row,flush=True)
    adapter.eval();torch.save({"adapter":adapter.state_dict(),"q0_sha256":pp["checkpoint"]["sha256"],
                               "steps":args.steps},args.out/"adapter.pt")
    baseline,preds=evaluate(exact,dev)
    results={"baseline":baseline,"history":history,"adapter_parameters":sum(p.numel() for p in adapter.parameters()),
             "policies":{},"promoted":False}
    for name,adaptive in (("periodic_ssm4",False),("adaptive_ssm4",True)):
        measured,preds=evaluate(HybridStream(exact,adapter,adaptive),dev)
        measured["quality_gate"]=quality_gate(baseline["metrics"],measured["metrics"])
        measured["speedup"]=baseline["mean_ms"]/measured["mean_ms"]
        measured["screen_pass"]=measured["quality_gate"]["pass"] and measured["mean_ms"]<=.95*baseline["mean_ms"]
        torch.save(preds,args.out/(name+"_predictions.pt"));results["policies"][name]=measured
        q.write(args.out/"results.json",results)
        print(name,measured["metrics"]["balanced"],"skip",measured["skip_fraction"],"mean_ms",measured["mean_ms"],
              "gate",measured["quality_gate"],flush=True)
    assert file_record(pp["checkpoint"]["path"])==pp["checkpoint"]


if __name__=="__main__":main()
