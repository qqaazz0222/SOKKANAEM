"""Independently check calibration thresholds, schedules, and skipped outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import cv2
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qschedule_study import padded,predict
from sokkanaem.protocol import file_record
from sokkanaem.qschedule import ScheduleRisk
from sokkanaem.qflow import gray_image,correspondence,warp_depth


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise FileExistsError('New output required')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    root=Path('work_dirs/qpreflow_20260909');pp=json.loads((root/'protocol.json').read_text())
    results=json.loads((root/'results.json').read_text());thresholds=json.loads((root/'thresholds.json').read_text())
    records=pp['code']+pp['development']+[pp[k] for k in ('checkpoint','parent_protocol','traces','order')]
    for r in records:assert file_record(r['path'])==r
    traces=torch.load(pp['traces']['path'],weights_only=False)
    for r in traces:r['x'][:,14:]=0
    hx,hy,hv=padded([r for r in traces if not r['fit']],'cuda');checks={}
    for mode in ('carry','reset','mlp'):
        model=ScheduleRisk('mlp' if mode=='mlp' else 'ssm').cuda().eval()
        model.load_state_dict(torch.load(root/(mode+'_risk.pt'),weights_only=False)['risk'],strict=True)
        scores=predict(model,hx,mode=='reset');eligible=hv.bool();eligible[:,0]=False
        values=torch.quantile(scores[eligible],scores.new_tensor([.25,.5,.75])).cpu()
        expected=torch.tensor([thresholds[mode][str(n)] for n in (25,50,75)])
        assert torch.equal(values,expected)
        checks[mode]={'exact':True,'observations':int(eligible.sum())}
    audit={'source':file_record(__file__),'threshold_checks':checks,'data':{},'records':records,
           'inputs':[file_record(root/n) for n in ('protocol.json','results.json','thresholds.json','carry_risk.pt','reset_risk.pt','mlp_risk.pt')]}
    engine=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    for length,rec in zip(('L32','L256'),pp['development']):
        samples=torch.load(rec['path'],weights_only=False);stats={}
        for name,measured in results[length].items():
            if name in ('baseline','output_k2','output_k4'):continue
            preds=torch.load(root/(length+'_'+name+'_predictions.pt'),weights_only=False)
            audit['inputs'].append(file_record(root/(length+'_'+name+'_predictions.pt')))
            mode,quantile=name.split('_')[:2];threshold=thresholds[mode][quantile[1:]]
            counts=Counter();times={};elapsed=iter(measured['per_frame_ms']);replay=0;flows=0
            for ci,(sample,tele) in enumerate(zip(samples,measured['telemetry'])):
                age=0
                for t,info in enumerate(tele['frames']):
                    gap=0 if t==0 else age+1;assert info['gap_attempted']==gap
                    full=t==0 or gap==4 or (gap>=2 and info['predicted_risk']>threshold)
                    assert full==info['full_refresh']
                    expected_flow=2 if (0<gap<4 and (name.endswith('_late') or not full)) else 0
                    assert info['flow_calls']==expected_flow;flows+=expected_flow
                    age=0 if full else gap;assert info['cache_age']==age
                    reason=info['reason'];counts[reason]+=1;times.setdefault(reason,[]).append(next(elapsed))
                    if not full:
                        flow,reliable=correspondence(gray_image(sample['rgb'][t-1:t].cuda()),gray_image(sample['rgb'][t:t+1].cuda()),engine)
                        candidate,_=warp_depth(preds[ci][t-1:t].cuda(),flow,reliable,'nearest')
                        assert torch.equal(candidate.cpu(),preds[ci][t:t+1]);replay+=1
            stats[name]={'reasons':dict(counts),'flow_calculations':flows,'skip_replay_exact':replay,
                         'reason_mean_ms':{k:statistics.mean(v) for k,v in times.items()}}
        audit['data'][length]=stats
    for r in records+audit['inputs']:assert file_record(r['path'])==r
    audit['all_hashes_unchanged']=True
    audit['total_exact_skip_replay']=sum(v['skip_replay_exact'] for s in audit['data'].values() for v in s.values())
    args.out.mkdir(parents=True);q.write(args.out/'audit.json',audit)
    print('audit complete',audit['total_exact_skip_replay'],flush=True)


if __name__=='__main__':main()
