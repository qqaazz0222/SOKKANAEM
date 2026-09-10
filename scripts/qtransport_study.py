"""Expanded matched training: residual vs previous-depth candidate transport."""
import argparse
import copy
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact, MotionStream
from scripts.qstream_study import evaluate, quality_gate
from sokkanaem.protocol import file_record
from sokkanaem.qseparated import SeparatedDelta
from sokkanaem.qquality import depth_retention
from sokkanaem.qflat_state import flat_mask, flat_excess_loss
from sokkanaem.qstream import ChangeAwareQ0
from sokkanaem.qtemporal_refiner import TemporalErrorRefiner, TemporalRefinedStream, repair_target
from sokkanaem.qtransport import TransportRefiner, candidate_bank, candidate_target


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("residual", "transport"), required=True)
    ap.add_argument("--cache", type=Path); args = ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    root = Path("work_dirs/qquality_feature_20260908")
    pp = json.loads((root/"protocol.json").read_text())
    torch.set_num_threads(4); torch.manual_seed(736); torch.backends.cudnn.benchmark = False
    exact = load_exact(pp["checkpoint"])
    adapter = SeparatedDelta(True, True).cuda().eval().requires_grad_(False)
    adapter.load_state_dict(torch.load(root/"adapter.pt", map_location="cuda", weights_only=False)["adapter"])
    train_path = Path("work_dirs/architecture_readout_20260908/train_cache.pt")
    train = torch.load(train_path, weights_only=False)
    paths = [Path("work_dirs/qstream_20260908/development_data.pt"), Path("work_dirs/qquality_audit_20260908/long_development_data.pt")]
    dev = {name: torch.load(p, weights_only=False) for name,p in zip(("L32", "L256"), paths)}
    tf = {str(Path(p).resolve()) for s in train for pair in s["pairs"] for p in pair}
    df = {str(Path(p).resolve()) for ss in dev.values() for s in ss for pair in s["pairs"] for p in pair}
    assert not tf & df
    refiner = (TransportRefiner() if args.mode == "transport" else TemporalErrorRefiner(True)).cuda()
    args.out.mkdir(parents=True)
    protocol = {"role": "reused_development_no_final_no_promotion", "checkpoint": pp["checkpoint"],
                "adapter": file_record(root/"adapter.pt"), "training_data": file_record(train_path),
                "development": [file_record(p) for p in paths], "mode": args.mode,
                "input_cache": file_record(args.cache) if args.cache else None,
                "steps": 1200, "seed": 736, "lr": .0005, "training_clips": len(train),
                "training_frames": sum(len(s["rgb"]) for s in train), "parameters": sum(p.numel() for p in refiner.parameters()),
                "objective": "5Q0logdepth+10Q0loggradient+20GTflat_excess + .1BCErepair(residual) or .1CEcandidate(transport)",
                "comparison": "same expanded data/order/steps/base/backbone family; output representation and supervision differ, NOT parameter-identical single-factor control",
                "transport": "49 previous-depth offsets from(-4,-2,-1,0,1,2,4)^2 plus coarse; hard argmax, soft ST training surrogate; +/- .03 residual; no flow oracle",
                "target": "trainingQ0 nearest log-depth candidate only when advantage over coarse>.005, otherwise coarse",
                "base": "same pooled-held feature SSM K2; no recurrent h carry across refresh",
                "code": [file_record(p) for p in [__file__, "sokkanaem/qtransport.py", "sokkanaem/qtemporal_refiner.py",
                          "sokkanaem/qseparated.py", "sokkanaem/qflat_state.py", "sokkanaem/qquality.py", "sokkanaem/qdelta_ssm.py",
                          "sokkanaem/qstream.py", "scripts/qquality_study.py", "scripts/qstream_study.py",
                          "sokkanaem/sharpness.py", "scripts/quality_refinement.py"]]}
    q.write(args.out/"protocol.json", protocol)
    if args.cache:
        cache = torch.load(args.cache, weights_only=False)
        expected = {(tuple(s["pairs"][t]), tuple(s["pairs"][t-1])) for s in train for t in range(1,len(s["rgb"]),2)}
        assert {(tuple(s["pairs"]), tuple(s["previous_pairs"])) for s in cache} == expected
    else:
        cache = []
        with torch.no_grad():
            for i,s in enumerate(train):
                rgb = s["rgb"].cuda().float(); teacher = exact.decode(exact.encode(rgb))
                flat,valid = flat_mask(s["gt"].cuda(),s["valid"].cuda())
                stream = MotionStream(exact,adapter,refresh_every=2); state = None; previous_depth = None
                for t in range(len(rgb)):
                    coarse,state,info = stream.step(rgb[t:t+1],state)
                    if not info["full_refresh"]:
                        cache.append({"current":rgb[t:t+1].cpu(),"previous":rgb[t-1:t].cpu(),"coarse":coarse.cpu(),
                                      "previous_depth":previous_depth.cpu(),"teacher":teacher[t:t+1].cpu(),
                                      "flat":flat[t:t+1].cpu(),"valid":valid[t:t+1].cpu(),
                                      "repair":repair_target(coarse,teacher[t:t+1]).cpu(),
                                      "candidate":candidate_target(candidate_bank(coarse,previous_depth),teacher[t:t+1]).cpu(),
                                      "pairs":s["pairs"][t],"previous_pairs":s["pairs"][t-1],"source":s["source"]})
                    previous_depth = coarse
                if (i+1)%48 == 0: print("cache clips",i+1,"/",len(train),flush=True)
        torch.save(cache,args.out/"training_cache.pt")
    assert len(cache)==sum(len(s["rgb"])//2 for s in train)
    opt = torch.optim.AdamW(refiner.parameters(),lr=.0005)
    gen = torch.Generator().manual_seed(736); history = []
    for step in range(1,1201):
        idx = torch.randint(len(cache),(4,),generator=gen).tolist()
        b = {k:torch.cat([cache[i][k] for i in idx]).cuda() for k in
             ("current","previous","coarse","previous_depth","teacher","flat","valid","repair","candidate")}
        pred,logits = refiner(b["current"],b["previous"],b["coarse"],b["previous_depth"])
        pred = pred.clamp(min=exact.q.d_min,max=exact.q.d_max)
        depth,edge = depth_retention(pred,b["teacher"])
        flat = flat_excess_loss(pred,b["teacher"],b["valid"],b["flat"])
        auxiliary = F.cross_entropy(logits,b["candidate"]) if args.mode=="transport" else F.binary_cross_entropy_with_logits(logits,b["repair"])
        loss = 5*depth+10*edge+20*flat+.1*auxiliary
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%200==0:
            row={"step":step,"loss":float(loss.detach()),"components":[float(x.detach()) for x in (depth,edge,flat,auxiliary)]}
            history.append(row);print(row,flush=True)
    refiner.eval();torch.save({"refiner":refiner.state_dict(),"mode":args.mode},args.out/"refiner.pt")
    q.write(args.out/"history.json",history);result={"promoted":False}
    for length,samples in dev.items():
        baseline,_=evaluate(exact,samples);result[length]={"baseline":baseline}
        kwargs={"d_min":exact.q.d_min,"d_max":exact.q.d_max}
        models={"ssm_base":MotionStream(exact,adapter,refresh_every=2),
                "ssm_repair":TemporalRefinedStream(MotionStream(exact,adapter,refresh_every=2),refiner,**kwargs),
                "hold_repair":TemporalRefinedStream(ChangeAwareQ0(exact,refresh_every=2,periodic_only=True),refiner,**kwargs)}
        if args.mode=="transport":
            for selection in ("soft","disabled"):
                m=copy.deepcopy(refiner);m.selection=selection
                models[selection]=TemporalRefinedStream(MotionStream(exact,adapter,refresh_every=2),m,**kwargs)
        for name,model in models.items():
            measured,preds=evaluate(model,samples)
            measured["quality_gate"]=quality_gate(baseline["metrics"],measured["metrics"])
            measured["speedup"]=baseline["mean_ms"]/measured["mean_ms"]
            measured["screen_pass"]=measured["quality_gate"]["pass"] and measured["speedup"]>=1/.95
            result[length][name]=measured
            torch.save(preds,args.out/(length+"_"+name+"_predictions.pt"));q.write(args.out/"results.json",result)
            print(length,name,measured["metrics"]["balanced"],"ms",measured["mean_ms"],measured["quality_gate"],flush=True)
    for r in protocol["code"]+[protocol["checkpoint"],protocol["adapter"]]:assert file_record(r["path"])==r


if __name__=="__main__":main()
