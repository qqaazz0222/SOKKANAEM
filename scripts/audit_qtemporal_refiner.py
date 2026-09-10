"""Saved-output parity, gate diagnostics, and an explicitly non-deployable oracle."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qstream_study import quality_gate
from sokkanaem.protocol import file_record
from sokkanaem.qtemporal_refiner import TemporalErrorRefiner, repair_target


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
    if args.out.exists(): raise FileExistsError("New output required")
    torch.set_num_threads(4)
    runs={k:Path("work_dirs/qtemporal_"+k+"_20260909") for k in ("current","pair")}
    result={"source":file_record(__file__),"role":"posthoc_diagnostics; oracle uses denseQ0 on skipped frames and is NOT a deployable model", "lengths":{}}
    for root in runs.values():
        p=json.loads((root/"protocol.json").read_text())
        for r in p["code"]+[p["checkpoint"],p["adapter"],p["train_cache"],p["training_gt"]]+p["development"]:
            assert file_record(r["path"])==r
    for length,data_path in (("L32","work_dirs/qstream_20260908/development_data.pt"),
                             ("L256","work_dirs/qquality_audit_20260908/long_development_data.pt")):
        samples=torch.load(data_path,weights_only=False)
        base=torch.load(runs["pair"]/(length+"_ssm_base_predictions.pt"),weights_only=False)
        teacher=torch.load("work_dirs/qquality_audit_20260908/"+length+"_dense_predictions.pt",weights_only=False)
        result["lengths"][length]={}; oracle_rows=[]; required=0; pixels=0
        for s,b,t in zip(samples,base,teacher):
            ratio=(t.clamp_min(1e-4)/b.clamp_min(1e-4)).log()
            required+=int((ratio[1::2].abs()>.1).sum()); pixels+=ratio[1::2].numel()
            oracle=b*ratio.clamp(-.1,.1).exp(); oracle[::2]=t[::2]
            oracle_rows.append({"source":s["source"],"scene":s["scene"],"metrics":q.score(oracle.cuda(),s)})
        baseline=json.loads((runs["pair"]/"results.json").read_text())[length]["baseline"]["metrics"]
        metrics=q.aggregate(oracle_rows)
        result["lengths"][length]["oracle"]={"metrics":metrics,"gate":quality_gate(baseline,metrics),
            "fraction_needing_over_0.1_log_change":required/pixels,
            "note":"Pixelwise closest Q0 target within log box, NOT a theorem about best GT-gradient metrics; no speed assigned"}
        for key,root in runs.items():
            ckpt=torch.load(root/"refiner.pt",map_location="cuda",weights_only=False)
            model=TemporalErrorRefiner(ckpt["temporal"]).cuda().eval();model.load_state_dict(ckpt["refiner"])
            actual=torch.load(root/(length+"_ssm_repair_predictions.pt"),weights_only=False)
            total=0;tp=0;fp=0;fn=0;brier=0.;mean_gate=0.;refresh_count=0
            for s,b,t,p in zip(samples,base,teacher,actual):
                torch.testing.assert_close(p[::2],t[::2],rtol=0,atol=0);refresh_count+=len(p[::2])
                for start in range(0,len(b),32):
                    cur=slice(start+1,min(start+32,len(b)),2);prev=slice(start,min(start+31,len(b)-1),2)
                    _,logit=model(s["rgb"][cur].cuda(),s["rgb"][prev].cuda(),b[cur].cuda(),t[prev].cuda())
                    truth=repair_target(b[cur].cuda(),t[cur].cuda());prob=logit.sigmoid();yes=prob>=.5
                    tp+=int((yes & truth.bool()).sum());fp+=int((yes & ~truth.bool()).sum());fn+=int((~yes & truth.bool()).sum())
                    total+=truth.numel();brier+=float((prob-truth).square().sum());mean_gate+=float(prob.sum())
            result["lengths"][length][key]={"exact_refresh_frames":refresh_count,"gate_precision":tp/max(tp+fp,1),
                "gate_recall":tp/max(tp+fn,1),"gate_brier":brier/total,"mean_gate":mean_gate/total,
                "note":"Pixel-weighted Q0-error classification on skipped frames; not GT-boundary precision or calibrated uncertainty"}
        print(length,result["lengths"][length],flush=True)
    args.out.mkdir(parents=True);q.write(args.out/"audit.json",result)


if __name__=="__main__":main()
