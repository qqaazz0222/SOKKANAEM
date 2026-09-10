"""Post-selection timing and one-frame initialization shift; no new thresholds."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import cv2
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact
from scripts.qquality_audit import latency_audit
from scripts.qstream_study import evaluate,quality_gate
from sokkanaem.protocol import file_record
from sokkanaem.qschedule import ScheduleRisk
from sokkanaem.qpreflow import PreflowScheduledFlow
from sokkanaem.qflow import RGBFlowStream


class ShiftedPreflow(PreflowScheduledFlow):
    @torch.no_grad()
    def step(self,frame,state=None):
        n=0 if state is None else state['n']
        if n==1:state=None
        pred,state,info=super().step(frame,state);state['n']=n+1
        return pred,state,info


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise FileExistsError('New output required')
    root=Path('work_dirs/qpreflow_20260909');pp=json.loads((root/'protocol.json').read_text())
    result=json.loads((root/'results.json').read_text());thresholds=json.loads((root/'thresholds.json').read_text())
    records=pp['code']+pp['development']+[pp[k] for k in ('checkpoint','parent_protocol','traces','order')]
    records += [file_record(root/n) for n in ('results.json','thresholds.json','mlp_risk.pt')]
    records += [file_record(__file__),file_record('scripts/qquality_audit.py')]
    for r in records:assert file_record(r['path'])==r
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    exact=load_exact(pp['checkpoint']);risk=ScheduleRisk('mlp').cuda().eval().requires_grad_(False)
    risk.load_state_dict(torch.load(root/'mlp_risk.pt',weights_only=False)['risk'],strict=True)
    threshold=thresholds['mlp']['50'];args.out.mkdir(parents=True)
    q.write(args.out/'protocol.json',{'role':'post_selection_reused_development_not_independent_validation','records':records,
        'candidate':'mlp_q50; unchanged held-prediction median; selected after initial development screen',
        'phase':'first two frames exact, then unchanged policy; no tuning',
        'timing':'three order-rotated repeats,64warmup,shared RTX4090, synchronized GPUresident including internal flow/transfers; no edge claim'})
    audit={}
    for length,rec in zip(('L32','L256'),pp['development']):
        samples=torch.load(rec['path'],weights_only=False);dense=torch.load(root/(length+'_dense_predictions.pt'),weights_only=False)
        phase,preds=evaluate(ShiftedPreflow(exact,risk,threshold),samples)
        phase['quality_gate']=quality_gate(result[length]['baseline']['metrics'],phase['metrics']);count=0
        for ci,tele in enumerate(phase['telemetry']):
            for t,info in enumerate(tele['frames']):
                if info['full_refresh']:assert torch.equal(preds[ci][t],dense[ci][t]);count+=1
        phase['exact_refresh_comparisons']=count;torch.save(preds,args.out/(length+'_shifted_predictions.pt'))
        print(length,'shifted',phase['quality_gate'],flush=True)
        timings=latency_audit({'dense':exact,'output_k2':RGBFlowStream(exact),
            'mlp_q50':PreflowScheduledFlow(exact,risk,threshold),'mlp_q50_late':PreflowScheduledFlow(exact,risk,threshold,late=True)},samples)
        means={k:statistics.mean(v['mean_ms'] for v in rows) for k,rows in timings.items()}
        audit[length]={'shifted':phase,'timings':timings,'means_ms':means,
            'speedup_vs_dense':means['dense']/means['mlp_q50'],'speedup_vs_k2':means['output_k2']/means['mlp_q50']}
        q.write(args.out/'audit.json',audit);print(length,means,flush=True)
    for r in records:assert file_record(r['path'])==r
    q.write(args.out/'integrity.json',{'hashes_unchanged':True})


if __name__=='__main__':main()
