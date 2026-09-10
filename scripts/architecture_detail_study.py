"""Stage 2 independent hypothesis: RGB feature pathway, ORIGINAL frozen head.

Executed only as an independent ablation if stage 1 fails; never combines a
rejected readout. Uses the stage-1 data without selecting new favorable clips.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import architecture_readout_study as study
from scripts import quality_refinement as q
from sokkanaem.detail_feature_path import DetailFeaturePath
from sokkanaem.mixture_readout import supervised_loss
from sokkanaem.model import from_checkpoint
from sokkanaem.protocol import file_record, check_final_test


@torch.no_grad()
def evaluate(samples,path,head):
    rows=[]
    for s in samples:
        p=head(path(s["rgb"].cuda().float(),s["features"].cuda().float()))
        rows.append({"source":s["source"],"scene":s["scene"],"metrics":q.score(p,s)})
    return q.aggregate(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--parent",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True);ap.add_argument("--steps",type=int,default=1200)
    args=ap.parse_args()
    if args.out.exists(): raise FileExistsError("Immutable experiment directory exists")
    p=json.loads((args.parent/"protocol.json").read_text())
    assert check_final_test(p["manifest"]["path"])=="development_validation"
    assert file_record(p["checkpoint"]["path"])==p["checkpoint"]
    args.out.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(p["seed"]);torch.backends.cudnn.benchmark=False
    train=torch.load(args.parent/"train_cache.pt",weights_only=False)
    dev=torch.load(args.parent/"development_cache.pt",weights_only=False)
    base=from_checkpoint(p["checkpoint"]["path"],"cuda").eval().requires_grad_(False)
    head=study.BinControl(base.decoder).cuda().eval().requires_grad_(False);del base
    before={k:v.detach().clone() for k,v in head.state_dict().items()}
    path=DetailFeaturePath().cuda()
    protocol={"role":"adaptive_development_stage2_independent_of_rejected_mixture",
              "parent":file_record(args.parent/"protocol.json"),"checkpoint":p["checkpoint"],
              "train_cache":file_record(args.parent/"train_cache.pt"),
              "dev_cache":file_record(args.parent/"development_cache.pt"),
              "seed":p["seed"],"steps":args.steps,"lr":.0001,"batch":8,
              "trainable":"only RGB half/quarter feature pathway and feature fusion",
              "frozen":"all original backbone, decoder and bin head",
              "code":[file_record(x) for x in [__file__,"sokkanaem/detail_feature_path.py",
                         "sokkanaem/mixture_readout.py","scripts/architecture_readout_study.py"]]}
    q.write(args.out/"protocol.json",protocol)
    baseline=study.evaluate(dev);initial=evaluate(dev,path,head)
    data={k:torch.cat([s[k] for s in train]).cuda().float() for k in ("rgb","features","gt","valid")}
    opt=torch.optim.AdamW(path.parameters(),lr=.0001,weight_decay=.0001)
    gen=torch.Generator(device="cuda").manual_seed(p["seed"]);history=[]
    for step in range(1,args.steps+1):
        idx=torch.randint(len(data["gt"]),(8,),device="cuda",generator=gen)
        x={k:v[idx] for k,v in data.items()}
        pred=head(path(x["rgb"],x["features"]))
        loss,parts=supervised_loss(pred,x["gt"],x["valid"])
        if not torch.isfinite(loss): raise ValueError("Nonfinite loss")
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(path.parameters(),1.);opt.step()
        if step==1 or step%200==0:
            row={"step":step,"loss":float(loss.detach()),**parts};history.append(row)
            print("detail",row,flush=True)
    path.eval();metrics=evaluate(dev,path,head)
    assert all(torch.equal(v,head.state_dict()[k]) for k,v in before.items())
    assert file_record(p["checkpoint"]["path"])==p["checkpoint"]
    torch.save({"detail_path":path.state_dict(),"base_sha256":p["checkpoint"]["sha256"],
                "steps":args.steps},args.out/"detail_path.pt")
    result={"baseline":baseline,"initial":initial,"candidate":metrics,"gate":study.strict_gate(baseline,metrics),
            "history":history,"parameters":sum(x.numel() for x in path.parameters()),
            "original_head_unchanged":True,"promoted":False}
    q.write(args.out/"results.json",result)
    print(metrics["balanced"],result["gate"],flush=True)


if __name__=="__main__":main()
