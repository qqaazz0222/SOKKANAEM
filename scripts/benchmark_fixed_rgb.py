"""RGB-only fixed-camera pilot: timing/reference retention, NOT GT accuracy."""
import json
from pathlib import Path
import statistics
import time
import cv2
import numpy as np
from PIL import Image
import torch
from scripts.acquire_fixed_camera_sbm import ROOT
from scripts.qquality_study import load_exact
from sokkanaem.data import ClipDataset
from sokkanaem.protocol import file_record, runtime_versions
from sokkanaem.qfixed_controls import PeriodicOutputHold
from sokkanaem.qflow import RGBFlowStream
from sokkanaem.qpreflow import PreflowScheduledFlow
from sokkanaem.qschedule import ScheduleRisk


@torch.no_grad()
def main():
    out=ROOT/'rgb_only_benchmark'
    if out.exists(): raise FileExistsError('Fresh experiment directory required')
    admission=json.loads((ROOT/'admission.json').read_text())
    candidate_path=Path('paper/streaming_draft/frozen_candidate.json')
    candidate=json.loads(candidate_path.read_text())
    records=candidate['weights']+candidate['code']+candidate['evidence']
    for r in records: assert file_record(r['path'])==r
    rgb_records=[file_record(r) for _,r,_ in admission['pairs']]
    assert all(x['pixel_exact'] for x in admission['raw_rgb_prefix_matches'])
    assert not admission['admitted_for_single_event_pilot']  # preserve the failed GT admission
    out.mkdir()
    sources=[__file__,'sokkanaem/qfixed_controls.py','sokkanaem/data.py']
    protocol={'source':[file_record(p) for p in sources],'candidate':file_record(candidate_path),
              'admission':file_record(ROOT/'admission.json'),'rgb_inputs':rgb_records,
              'frames':160,'resets':'once at the start of each complete sequence, no event concatenation',
              'repeats':3,'method_order':['dense','hold_k2','flow_k2','mlp_q50'],
              'model_order_note':'fixed order; no randomized crossover; shared-device pilot',
              'masks':'7 provided foreground annotations; foreground includes stopped objects, not instantaneous motion',
              'reference':'Q0 predictions only; no sensor GT accuracy or quality-gate decision',
              'admitted_rgb_only':True,'gt_admission_still_failed':True,
              'timing':'resident-input synchronized invocation; RGB loading and initial H2D excluded; internal CPU flow included; not end-to-end/device realtime',
              'frozen_records':records,'runtime':runtime_versions()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.set_num_threads(4); cv2.setNumThreads(1); torch.backends.cudnn.benchmark=False
    loader=ClipDataset([],1000,clip_len=160,size=256,strict=True)
    rgb=torch.stack([loader._rgb(r['path']) for r in rgb_records])
    torch.save(rgb,out/'rgb.pt')
    exact=load_exact(candidate['weights'][1])
    risk=ScheduleRisk('mlp').cuda().eval().requires_grad_(False)
    risk.load_state_dict(torch.load(candidate['weights'][0]['path'],weights_only=False)['risk'],strict=True)
    models={'dense':exact,'hold_k2':PeriodicOutputHold(exact,2),'flow_k2':RGBFlowStream(exact),
            'mlp_q50':PreflowScheduledFlow(exact,risk,candidate['threshold'])}
    results={}; saved={}; parity=0; replay=0
    for repeat in range(3):
        for name,model in models.items():
            frame=rgb[:1].cuda(); state=None
            for _ in range(16): _,state,_=model.step(frame,state)
            torch.cuda.synchronize(); state=None; predictions=[]; elapsed=[]; telemetry=[]
            for x in rgb:
                frame=x[None].cuda(); torch.cuda.synchronize(); start=time.perf_counter()
                pred,state,info=model.step(frame,state); torch.cuda.synchronize()
                elapsed.append((time.perf_counter()-start)*1000)
                predictions.append(pred.cpu()); telemetry.append(info)
            pred=torch.cat(predictions)
            if repeat==0:
                saved[name]=pred; torch.save(pred,out/f'{name}_predictions.pt')
            else:
                assert torch.equal(pred,saved[name]); replay+=len(pred)
            if name!='dense':
                for t,info in enumerate(telemetry):
                    if info['full_refresh']:
                        assert torch.equal(pred[t],saved['dense'][t]); parity+=1
            row={'mean_ms':statistics.mean(elapsed),'p95_ms':float(np.quantile(elapsed,.95)),
                 'per_frame_ms':elapsed,'telemetry':telemetry,
                 'full_calls':sum(i['backbone_calls'] for i in telemetry)}
            results.setdefault(name,{'runs':[]})['runs'].append(row)
            print('TIMING',repeat,name,row['mean_ms'],row['p95_ms'],'calls',row['full_calls'],flush=True)
    labels={int(i):torch.from_numpy(np.array(loader._fit(Image.open(p),Image.Resampling.NEAREST)).copy())
            for i,p in admission['masks'].items()}
    baseline=saved['dense']
    for name,result in results.items():
        result['mean_of_run_means_ms']=statistics.mean(r['mean_ms'] for r in result['runs'])
        result['skip_fraction']=1-result['runs'][0]['full_calls']/160
        delta=(saved[name]-baseline).abs()/baseline.clamp_min(1e-6)
        result['reference_relative_difference_mean']=float(delta.mean())
        regions={}
        for label,title in [(0,'annotated_background'),(255,'annotated_foreground')]:
            values=[]; frames=[]
            for i,mask in labels.items():
                keep=mask==label
                if keep.any():
                    values.append(delta[i-1,0][keep]);frames.append(i)
            v=torch.cat(values) if values else None
            regions[title]={'frame_ids':frames,'pixels':0 if v is None else v.numel(),
                            'reference_relative_difference_mean':None if v is None else float(v.mean()),
                            'not_depth_gt_accuracy':True}
        result['annotated_regions']=regions
    for r in records+protocol['source']+rgb_records+[protocol['candidate'],protocol['admission']]:
        assert file_record(r['path'])==r
    final={'protocol':file_record(out/'protocol.json'),'results':results,
           'exact_refresh_comparisons':parity,'exact_repeated_frames':replay,
           'gt_quality_gate_evaluated':False,'candidate_promoted':False,'hashes_unchanged':True,
           'gpu':torch.cuda.get_device_name(),'predictions':[file_record(out/f'{n}_predictions.pt') for n in models]}
    (out/'results.json').write_text(json.dumps(final,indent=2)+'\n')
    print('COMPLETE RGB ONLY',parity,replay,flush=True)


if __name__=='__main__':main()
