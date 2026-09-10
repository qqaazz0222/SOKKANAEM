"""Candidate usage and teacher-reference diagnostics, not deployable oracle results."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qstream_study import quality_gate
from sokkanaem.protocol import file_record
from sokkanaem.qtransport import TransportRefiner, candidate_bank, candidate_target, OFFSETS


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise FileExistsError("New output required")
    torch.set_num_threads(4)
    roots=[Path("work_dirs/qtransport_"+x+"_20260909") for x in ("residual","hard")]
    for root in roots:
        p=json.loads((root/"protocol.json").read_text())
        for r in p["code"]+[p["checkpoint"],p["adapter"],p["training_data"]]+p["development"]:
            assert file_record(r["path"])==r
        if p["input_cache"]:assert file_record(p["input_cache"]["path"])==p["input_cache"]
    cache=torch.load(roots[0]/"training_cache.pt",weights_only=False)
    model=TransportRefiner().cuda().eval()
    model.load_state_dict(torch.load(roots[1]/"refiner.pt",map_location="cuda",weights_only=False)["refiner"])
    result={"source":file_record(__file__),"role":"posthoc_reused_development_diagnostics; oracle unavailable in deployment",
            "training_candidates":len(cache),"training_noncoarse_target_fraction":sum(int((s["candidate"]>0).sum()) for s in cache)/sum(s["candidate"].numel() for s in cache),"data":{}}
    for length,path in (("L32","work_dirs/qstream_20260908/development_data.pt"),("L256","work_dirs/qquality_audit_20260908/long_development_data.pt")):
        data=torch.load(path,weights_only=False)
        base=torch.load(roots[1]/(length+"_ssm_base_predictions.pt"),weights_only=False)
        teacher=torch.load("work_dirs/qquality_audit_20260908/"+length+"_dense_predictions.pt",weights_only=False)
        actual=torch.load(roots[1]/(length+"_ssm_repair_predictions.pt"),weights_only=False)
        histogram=torch.zeros(50,dtype=torch.long);right=0;total=0;oracle_rows=[];refresh=0
        for s,b,t,pred in zip(data,base,teacher,actual):
            torch.testing.assert_close(pred[::2],t[::2],rtol=0,atol=0);refresh+=len(pred[::2])
            oracle=b.clone()
            for start in range(0,len(b),32):
                cur=slice(start+1,min(start+32,len(b)),2);prev=slice(start,min(start+31,len(b)-1),2)
                bank=candidate_bank(b[cur].cuda(),t[prev].cuda());labels=candidate_target(bank,t[cur].cuda())
                _,logits=model(s["rgb"][cur].cuda(),s["rgb"][prev].cuda(),b[cur].cuda(),t[prev].cuda())
                selected=logits.argmax(1)
                histogram+=torch.bincount(selected.flatten(),minlength=50).cpu()
                right+=int((selected==labels).sum());total+=labels.numel()
                oracle[cur]=bank.gather(1,labels[:,None]).cpu()
            oracle_rows.append({"source":s["source"],"scene":s["scene"],"metrics":q.score(oracle.cuda(),s)})
        metrics=q.aggregate(oracle_rows)
        baseline=json.loads((roots[1]/"results.json").read_text())[length]["baseline"]["metrics"]
        zero_index=1+OFFSETS.index(0)*len(OFFSETS)+OFFSETS.index(0)
        row={"exact_refresh_frames":refresh,"selected_noncoarse_fraction":1-float(histogram[0])/total,
             "selected_nonzero_offset_fraction":1-float(histogram[0]+histogram[zero_index])/total,
             "teacher_candidate_top1":right/total,"histogram":histogram.tolist(),"oracle_metrics":metrics,
             "oracle_gate":quality_gate(baseline,metrics),
             "oracle_note":"Training target rule applied using current denseQ0; not GT-optimal or achievable without teacher, no speed claim"}
        result["data"][length]=row
        print(length,{k:v for k,v in row.items() if k not in ("histogram","oracle_metrics")},"oracle",metrics["balanced"],flush=True)
    args.out.mkdir(parents=True);q.write(args.out/"audit.json",result)


if __name__=="__main__":main()
