"""Sequential stage 1: head-only control versus anchored bimodal readout.

Keeps original backbone, decoder features and paper artifacts immutable.
"""
import argparse
import copy
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from sokkanaem.model import from_checkpoint, checkpoint_config
from sokkanaem.data import build_mixed, load_manifest
from sokkanaem.protocol import file_record, check_final_test
from sokkanaem.mixture_readout import AnchoredMixtureReadout, supervised_loss


class BinControl(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.head=copy.deepcopy(decoder.head).requires_grad_(True)
        self.bin_logits=torch.nn.Parameter(decoder.bin_logits.detach().clone())
        self.register_buffer("log_range",decoder.log_range.detach().clone())
    def forward(self,features):
        lo,hi=self.log_range
        widths=self.bin_logits.softmax(0)*(hi-lo)
        centers=lo+widths.cumsum(0)-widths/2
        logd=(self.head(features).softmax(1)*centers[None,:,None,None]).sum(1,keepdim=True)
        return F.interpolate(logd,scale_factor=2,mode="bilinear",align_corners=False).exp()


@torch.no_grad()
def evaluate(samples, model=None, variant="base"):
    rows=[]
    for s in samples:
        c=s["coarse"].cuda(); f=s["features"].cuda().float()
        p=c if model is None else (model(f) if variant=="bin" else model(f,c)[variant])
        rows.append({"source":s["source"],"scene":s["scene"],"metrics":q.score(p,s)})
    return q.aggregate(rows)


def strict_gate(base, result):
    out=q.gate(base,result)
    out["checks"]["edge_depth_accuracy_nonregression"] = result["balanced"]["absrel_edge"] <= base["balanced"]["absrel_edge"]
    out["pass"]=all(out["checks"].values())
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--steps",type=int,default=1200)
    args=ap.parse_args()
    if args.out.exists(): raise FileExistsError("Immutable output directory exists")
    parent=Path("work_dirs/quality_refinement_20260908")
    pp=json.loads((parent/"protocol.json").read_text())
    assert file_record(pp["checkpoint"]["path"])==pp["checkpoint"]
    assert check_final_test(pp["manifest"]["path"])=="development_validation"
    torch.set_num_threads(4); torch.manual_seed(20260909)
    torch.backends.cudnn.benchmark=False
    cfg=checkpoint_config(pp["checkpoint"]["path"])
    old=json.loads((parent/"selection.json").read_text())
    excluded={str(Path(p).resolve()) for c in old["development"] for pair in c["pairs"] for p in pair}
    dev_select=[]
    for src,ds in load_manifest(pp["manifest"]["path"]):
        groups=defaultdict(list)
        for i in range(len(ds)):
            pairs=q.paths(ds,i)
            if not any(str(Path(p).resolve()) in excluded for pair in pairs for p in pair):
                groups[q.scene(pairs[0])].append(i)
        for name,indices in groups.items():
            if len(indices)<4: raise ValueError("Insufficient disjoint development clips")
            for j in np.linspace(0,len(indices)-1,4).astype(int):
                dev_select.append((src,ds,indices[j]))
    assert len(dev_select)==16
    specs=[s for s in cfg["data"] if s.split(":")[0] in ("tum","bonn","vkitti2")]
    train,_=build_mixed(specs,holdout=cfg["holdout"],clip_len=4,clip_stride=4,size=256,strict=True,augment=False)
    rng=np.random.default_rng(20260909); train_select=[]
    for spec,ds in zip(specs,train.datasets):
        for i in rng.choice(len(ds),64,replace=False): train_select.append((spec.split(":")[0],ds,int(i)))
    tf={str(Path(p).resolve()) for _,ds,i in train_select for pair in q.paths(ds,i) for p in pair}
    df={str(Path(p).resolve()) for _,ds,i in dev_select for pair in q.paths(ds,i) for p in pair}
    assert not tf & df and not df & excluded
    args.out.mkdir(parents=True)
    selection={k:[{"source":s,"pairs":q.paths(d,i)} for s,d,i in selected]
               for k,selected in (("train",train_select),("development",dev_select))}
    q.write(args.out/"selection.json",selection)
    protocol={"role":"new_frames_existing_development_scenes_not_final",
              "seed":20260909,"steps_per_arm":args.steps,"batch":8,"lr":.0001,
              "train_clips":192,"train_frames":768,"development_clips":16,"development_frames":128,
              "arms":["bin_head_finetune","anchored_mixture_mean","anchored_mixture_mode"],
              "note":"Mixture mean and mode are TWO READOUTS OF ONE CHECKPOINT; original bin anchor stays frozen. Mixture adds 0.05 log-depth NLL to common metric/gradient objective. Not a pure loss-controlled comparison to bin finetune.",
              "checkpoint":pp["checkpoint"],"manifest":pp["manifest"],
              "gate":"prior strict gate plus edge-region AbsRel nonregression; all required before stage 2",
              "selection":file_record(args.out/"selection.json"),
              "code":[file_record(p) for p in [__file__,"sokkanaem/mixture_readout.py","scripts/quality_refinement.py",
                          "sokkanaem/model.py","sokkanaem/data.py","sokkanaem/sharpness.py","sokkanaem/alignment.py"]]}
    q.write(args.out/"protocol.json",protocol)
    base=from_checkpoint(pp["checkpoint"]["path"],"cuda").eval().requires_grad_(False)
    binmodel=BinControl(base.decoder).cuda()
    caches=[]
    for name,selected in (("train",train_select),("development",dev_select)):
        samples=[]
        for j,(src,ds,i) in enumerate(selected):
            samples.append(q.extract(base,ds,i,src))
            if (j+1)%16==0: print("cache",name,j+1,"/",len(selected),flush=True)
        torch.save(samples,args.out/(name+"_cache.pt")); caches.append(samples)
    del base
    train,dev=caches
    baseline=evaluate(dev)
    initial_bin=evaluate(dev,binmodel,"bin")
    results={"baseline":baseline,"initial_bin_cached_features":initial_bin,"candidates":{},"promoted":False}
    q.write(args.out/"baseline.json",baseline)
    print("BASELINE",baseline["balanced"],flush=True)
    data={k:torch.cat([s[k] for s in train]).cuda().float() for k in ("coarse","features","gt","valid")}
    for kind in ("bin","mixture"):
        torch.manual_seed(protocol["seed"])
        model=binmodel if kind=="bin" else AnchoredMixtureReadout().cuda()
        # Independent generator makes minibatch order identical across arms.
        gen=torch.Generator(device="cuda").manual_seed(protocol["seed"])
        opt=torch.optim.AdamW(model.parameters(),lr=protocol["lr"],weight_decay=.0001)
        history=[]
        for step in range(1,args.steps+1):
            idx=torch.randint(len(data["gt"]),(8,),device="cuda",generator=gen)
            x={k:v[idx] for k,v in data.items()}
            output=None if kind=="bin" else model(x["features"],x["coarse"])
            pred=model(x["features"]) if kind=="bin" else output["mode"]
            loss,parts=supervised_loss(pred,x["gt"],x["valid"],output)
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step==1 or step%200==0:
                row={"step":step,"loss":float(loss.detach()),**parts};history.append(row)
                print(kind,row,flush=True)
        model.eval()
        path=args.out/(kind+".pt")
        torch.save({"state_dict":model.state_dict(),"kind":kind,"base_sha256":pp["checkpoint"]["sha256"],
                    "steps":args.steps},path)
        for readout in (["bin"] if kind=="bin" else ["mean","mode"]):
            metrics=evaluate(dev,model,readout)
            outcome={"metrics":metrics,"gate":strict_gate(baseline,metrics),"history":history,
                     "checkpoint":file_record(path),"trainable_parameters":sum(p.numel() for p in model.parameters())}
            results["candidates"][readout]=outcome
            print(readout,metrics["balanced"],outcome["gate"],flush=True)
        q.write(args.out/"results.json",results)
    results["stage2_authorized_by_gate"]=any(x["gate"]["pass"] for x in results["candidates"].values())
    assert file_record(pp["checkpoint"]["path"])==pp["checkpoint"]
    q.write(args.out/"results.json",results)


if __name__=="__main__": main()
