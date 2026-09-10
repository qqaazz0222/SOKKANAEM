"""Actual correction magnitude, RGB-mask error coverage, nondeployable oracles."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qstream_study import quality_gate
from sokkanaem.protocol import file_record
from sokkanaem.qflow import gray_image,correspondence,warp_depth
from sokkanaem.qtemporal_refiner import repair_target


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise FileExistsError("New output required")
    torch.set_num_threads(4);cv2.setNumThreads(1)
    root=Path("work_dirs/qregion_20260909");pp=json.loads((root/"protocol.json").read_text())
    results=json.loads((root/"results.json").read_text())
    records=pp["code"]+pp["development"]+[pp["checkpoint"],pp["training_cache"],pp["parent_protocol"]]
    for r in records:assert file_record(r["path"])==r
    engine=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    audit={"role":"posthoc_reused_development_diagnostics_oracles_unavailable_at_inference", "source":file_record(__file__),"data":{},"inputs":[]}
    for length,rec in zip(("L32","L256"),pp["development"]):
        samples=torch.load(rec["path"],weights_only=False)
        names=("output_flow","dense","local_local","global_global")
        loaded={n:torch.load(root/(length+"_"+n+"_predictions.pt"),weights_only=False) for n in names}
        audit["inputs"] += [file_record(root/(length+"_"+n+"_predictions.pt")) for n in names]
        pixels=editable_pixels=errors=hits=outside_count=0
        amount={n:{"sum_abs_log":0.,"max_abs_log":0.,"changed_gt_1e_4":0,"changed_gt_1e_6":0} for n in ("local_local","global_global")}
        oracle_rows={"bounded":[],"support_only":[]}
        for ci,s in enumerate(samples):
            base=loaded["output_flow"][ci];teacher=loaded["dense"][ci]
            oracles={n:base.clone() for n in oracle_rows}
            for t in range(1,len(base),2):
                previous=s["rgb"][t-1:t].cuda();current=s["rgb"][t:t+1].cuda()
                flow,reliable=correspondence(gray_image(previous),gray_image(current),engine)
                warped,mask=warp_depth(teacher[t-1:t].cuda(),flow,reliable,"nearest")
                torch.testing.assert_close(warped.cpu(),base[t:t+1],rtol=0,atol=0)
                edit=~mask;truth=teacher[t:t+1].cuda()
                need=repair_target(warped,truth).bool()
                pixels+=edit.numel();editable_pixels+=int(edit.sum());errors+=int(need.sum());hits+=int((need&edit).sum())
                actual=loaded["local_local"][ci][t:t+1].cuda()
                assert torch.equal(actual[~edit],warped[~edit]);outside_count+=int((~edit).sum())
                for name in amount:
                    delta=(loaded[name][ci][t:t+1].cuda().log()-warped.log()).abs()
                    amount[name]["sum_abs_log"]+=float(delta.sum())
                    amount[name]["max_abs_log"]=max(amount[name]["max_abs_log"],float(delta.max()))
                    amount[name]["changed_gt_1e_4"]+=int((delta>1e-4).sum());amount[name]["changed_gt_1e_6"]+=int((delta>1e-6).sum())
                alpha=F.avg_pool2d(edit.float(),5,stride=1,padding=2)*edit
                target_delta=truth.log()-warped.log();bound=.03*alpha
                bounded=warped*target_delta.maximum(-bound).minimum(bound).exp()
                oracles["bounded"][t:t+1]=torch.where(edit,bounded,warped).cpu()
                oracles["support_only"][t:t+1]=torch.where(edit,truth,warped).cpu()
            for name,pred in oracles.items():
                oracle_rows[name].append({"source":s["source"],"scene":s["scene"],"metrics":q.score(pred.cuda(),s)})
        for value in amount.values():
            value["mean_abs_log"]=value.pop("sum_abs_log")/pixels
            value["fraction_gt_1e_4"]=value.pop("changed_gt_1e_4")/pixels
            value["fraction_gt_1e_6"]=value.pop("changed_gt_1e_6")/pixels
        oracle={}
        for name,rows in oracle_rows.items():
            metrics=q.aggregate(rows);oracle[name]={"metrics":metrics,"quality_gate":quality_gate(results[length]["baseline"]["metrics"],metrics)}
        row={"pixel_weighted_mask":{"editable_fraction":editable_pixels/pixels,"q0_error_proxy_precision":hits/max(1,editable_pixels),
                                     "q0_error_proxy_recall":hits/max(1,errors),"outside_exact_pixel_comparisons":outside_count},
             "correction":amount,"oracles":oracle}
        audit["data"][length]=row
        print(length,"mask",row["pixel_weighted_mask"],"correction",amount,flush=True)
        print(length,"oracles",{n:{"balanced":v["metrics"]["balanced"],"gate":v["quality_gate"]} for n,v in oracle.items()},flush=True)
    audit["notes"]=["Error proxy = Q0 logdepth error>.015 OR adjacent loggradient error>.02; NOT ground-truth occlusion/flow accuracy.",
                    "Bounded oracle projects current denseQ0 onto +/- .03*support-taper logdelta; support-only inserts current denseQ0 only inside mask. Both use unavailable teacher, no speed claims, not GT-optimal.",
                    "Pixel-weighted mask/magnitude diagnostics are not source/scene-balanced depth scores. Output equality does not imply normalized global metrics outside mask unchanged."]
    args.out.mkdir(parents=True);q.write(args.out/"audit.json",audit)
    for r in records:assert file_record(r["path"])==r


if __name__=="__main__":main()
