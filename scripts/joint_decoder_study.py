"""Toy learnability check followed by matched full-decoder/detail training.

No final test, no automatic promotion, no edits to existing model code.
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
from sokkanaem.model import from_checkpoint
from sokkanaem.joint_detail_decoder import JointDetailDecoder, reliable_edge_loss
from sokkanaem.mixture_readout import supervised_loss
from sokkanaem.protocol import file_record, check_final_test
from sokkanaem.sharpness import boundary_prf, grad_mag


def toy_boundary_f1(pred,gt):
    """Toy-only absolute log-gradient threshold; avoids zero GT quantiles.

    0.1 is well below the known log(4/2) step, but rejects tiny flat-region
    numerical changes. This is NOT the real development boundary metric.
    """
    valid=torch.ones_like(gt)
    gp,defined=grad_mag(pred.clamp_min(1e-6).log(),valid)
    gg,_=grad_mag(gt.log(),valid)
    bp=(gp>.1)&defined.bool();bg=(gg>.1)&defined.bool()
    grow=lambda x:F.max_pool2d(x.float(),5,1,2)>0
    precision=(bp&grow(bg)).flatten(1).sum(1)/bp.flatten(1).sum(1).clamp_min(1)
    recall=(bg&grow(bp)).flatten(1).sum(1)/bg.flatten(1).sum(1).clamp_min(1)
    return float((2*precision*recall/(precision+recall).clamp_min(1e-9)).mean())


def toy_data(seed, n, dim, blocks):
    gen=torch.Generator().manual_seed(seed)
    size=64
    yy,xx=torch.meshgrid(torch.arange(size),torch.arange(size),indexing="ij")
    masks=[]
    for _ in range(n):
        cx,cy=torch.randint(16,48,(2,),generator=gen)
        radius=int(torch.randint(8,20,(),generator=gen))
        masks.append(((xx-cx).square()+(yy-cy).square()<radius**2).float())
    mask=torch.stack(masks)[:,None]
    # Deliberately easy two-plane sanity task, NOT a realistic depth dataset.
    # Color is predictive of the two depths; independent fine texture is added.
    fg=torch.tensor([.2,.4,.8])[None,:,None,None]
    bg=torch.tensor([.8,.5,.2])[None,:,None,None]
    rgb=(fg*mask+bg*(1-mask)+.04*torch.randn(n,3,size,size,generator=gen)).clamp(0,1)
    return {"rgb":rgb.cuda(),"gt":(2*mask+4*(1-mask)).cuda(),
            "tokens":torch.zeros(n,blocks,dim,4,4,device="cuda")}


@torch.no_grad()
def toy_score(model, data):
    preds=[]
    for i in range(0,len(data["rgb"]),8):
        preds.append(model(data["tokens"][i:i+8].unbind(1),data["rgb"][i:i+8]))
    pred=torch.cat(preds);gt=data["gt"]
    return {"absrel":float(((pred-gt).abs()/gt).mean()),
            "boundary_f1":toy_boundary_f1(pred,gt),
            "legacy_quantile_boundary_f1":boundary_prf(pred,gt,torch.ones_like(gt))["f1"]}


def sanity(decoder,out):
    torch.manual_seed(731)
    model=JointDetailDecoder(decoder,True).cuda()
    train=toy_data(731,64,decoder.dim,len(decoder.proj))
    val=toy_data(732,32,decoder.dim,len(decoder.proj))
    initial=toy_score(model,val)
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    gen=torch.Generator(device="cuda").manual_seed(733)
    history=[]
    for step in range(1,501):
        ix=torch.randint(64,(8,),device="cuda",generator=gen)
        pred=model(train["tokens"][ix].unbind(1),train["rgb"][ix])
        gt=train["gt"][ix];v=torch.ones_like(gt)
        loss,_=supervised_loss(pred,gt,v)
        loss=loss+.1*reliable_edge_loss(pred,gt,v)
        if not torch.isfinite(loss):raise ValueError("nonfinite toy loss")
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0:
            history.append({"step":step,"loss":float(loss.detach())})
            print("sanity",history[-1],flush=True)
    final=toy_score(model,val)
    result={"initial":initial,"final":final,"history":history,
            "pass":final["absrel"]<.08 and final["boundary_f1"]>.70,
            "note":"Color-coded two-plane circles, independent train/validation seeds. Not real-depth accuracy or a backbone test. Toy weights NEVER transferred to real training."}
    q.write(out/"sanity.json",result);print("SANITY",result,flush=True)
    return result


@torch.no_grad()
def token_cache(base,samples):
    output=[];captured=[]
    hook=base.decoder.register_forward_pre_hook(lambda m,args:captured.append(torch.stack(args[0],1).detach().cpu()))
    try:
        for i,s in enumerate(samples):
            captured.clear();state=None
            for rgb in s["rgb"]:
                _,state,_=base.step(rgb[None].cuda().float(),state)
            assert len(captured)==len(s["rgb"])
            output.append({"rgb":s["rgb"],"gt":s["gt"],"valid":s["valid"],
                           "tokens":torch.cat(captured),"source":s["source"],
                           "scene":s["scene"],"pairs":s["pairs"]})
            if (i+1)%32==0:print("tokens",i+1,"/",len(samples),flush=True)
    finally:hook.remove()
    return output


@torch.no_grad()
def evaluate(model,samples):
    rows=[]
    for s in samples:
        p=model(s["tokens"].cuda().unbind(1),s["rgb"].cuda().float())
        rows.append({"source":s["source"],"scene":s["scene"],"metrics":q.score(p,s)})
    return q.aggregate(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--parent",type=Path,default=Path("work_dirs/architecture_readout_20260908"))
    ap.add_argument("--steps",type=int,default=1600)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("Use a fresh output directory")
    p=json.loads((args.parent/"protocol.json").read_text())
    assert check_final_test(p["manifest"]["path"])=="development_validation"
    assert file_record(p["checkpoint"]["path"])==p["checkpoint"]
    args.out.mkdir(parents=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    protocol={"role":"adaptive_joint_decoder_development_not_final","parent":file_record(args.parent/"protocol.json"),
              "checkpoint":p["checkpoint"],"steps_per_real_arm":args.steps,"seed":734,"lr":.00003,"batch":8,
              "supervision":"log L1 + 0.5 global signed-gradient L1 + 0.1 GT-discontinuity gradient L1, same for both arms",
              "sanity_gate":{"steps":500,"absrel_less_than":.08,"boundary_f1_greater_than":.70,
                             "toy_only_boundary_rule":"log-gradient > 0.1, 2px tolerance; legacy quantile F1 also retained"},
              "train_cache":file_record(args.parent/"train_cache.pt"),"dev_cache":file_record(args.parent/"development_cache.pt"),
              "code":[file_record(x) for x in [__file__,"sokkanaem/joint_detail_decoder.py","sokkanaem/detail_feature_path.py",
                         "sokkanaem/mixture_readout.py","scripts/quality_refinement.py","scripts/architecture_readout_study.py",
                         "sokkanaem/model.py","sokkanaem/data.py","sokkanaem/sharpness.py"]]}
    q.write(args.out/"protocol.json",protocol)
    base=from_checkpoint(p["checkpoint"]["path"],"cuda").eval().requires_grad_(False)
    result=sanity(base.decoder,args.out)
    if not result["pass"]:
        q.write(args.out/"results.json",{"sanity":result,"real_training_executed":False,"promoted":False})
        return
    train=token_cache(base,torch.load(args.parent/"train_cache.pt",weights_only=False))
    dev=token_cache(base,torch.load(args.parent/"development_cache.pt",weights_only=False))
    torch.save(train,args.out/"train_tokens.pt");torch.save(dev,args.out/"dev_tokens.pt")
    baseline=evaluate(JointDetailDecoder(base.decoder,False).cuda().eval(),dev)
    results={"sanity":result,"baseline":baseline,"arms":{},"promoted":False,"real_training_executed":True}
    data={k:torch.cat([s[k] for s in train]).cuda().float() for k in ("rgb","gt","valid","tokens")}
    for name,detail in (("decoder_only",False),("joint_detail",True)):
        torch.manual_seed(734);model=JointDetailDecoder(base.decoder,detail).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.00003,weight_decay=.0001)
        gen=torch.Generator(device="cuda").manual_seed(734);history=[]
        for step in range(1,args.steps+1):
            ix=torch.randint(len(data["gt"]),(8,),device="cuda",generator=gen)
            x={k:v[ix] for k,v in data.items()}
            pred=model(x["tokens"].unbind(1),x["rgb"])
            loss,parts=supervised_loss(pred,x["gt"],x["valid"])
            edge=reliable_edge_loss(pred,x["gt"],x["valid"]);loss=loss+.1*edge
            if not torch.isfinite(loss):raise ValueError("nonfinite real loss")
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step==1 or step%200==0:
                row={"step":step,"loss":float(loss.detach()),"edge":float(edge.detach()),**parts}
                history.append(row);print(name,row,flush=True)
        model.eval();metrics=evaluate(model,dev)
        torch.save({"state_dict":model.state_dict(),"detail":detail,"base_sha256":p["checkpoint"]["sha256"],
                    "steps":args.steps},args.out/(name+".pt"))
        outcome={"metrics":metrics,"gate":strict_gate(baseline,metrics),"history":history,
                 "parameters":sum(x.numel() for x in model.parameters())}
        results["arms"][name]=outcome;q.write(args.out/"results.json",results)
        print(name,metrics["balanced"],outcome["gate"],flush=True)
    assert file_record(p["checkpoint"]["path"])==p["checkpoint"]
    q.write(args.out/"results.json",results)


if __name__=="__main__":main()
