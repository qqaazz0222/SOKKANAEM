"""One-shot frozen-candidate evaluation on pre-registered new TUM sequences."""
import json
from pathlib import Path
import sys
import cv2
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import quality_refinement as q
from scripts.qquality_study import load_exact
from scripts.qstream_study import evaluate,quality_gate
from sokkanaem.data import load_manifest
from sokkanaem.protocol import file_record,check_final_test,runtime_versions
from sokkanaem.qschedule import ScheduleRisk
from sokkanaem.qpreflow import PreflowScheduledFlow
from sokkanaem.qflow import RGBFlowStream


def main():
    root=Path('work_dirs/qindependent_20260909');out=root/'evaluation'
    if out.exists():raise FileExistsError('Never overwrite new holdout predictions')
    prepared=json.loads((root/'prepared.json').read_text());pre=json.loads((root/'preregistered.json').read_text())
    candidate=json.loads(Path(pre['candidate']['path']).read_text())
    records=candidate['weights']+candidate['code']+candidate['evidence']+[pre['candidate'],prepared['preregistration'],prepared['frames']]+prepared['manifests']
    for r in records:assert file_record(r['path'])==r
    for r in json.loads((root/'frame_hashes.json').read_text()):assert file_record(r['path'])==r
    traces=torch.load('work_dirs/qschedule_20260909/traces.pt',weights_only=False)
    old_pairs=[p for t in traces for pair in t['pairs'] for p in pair]
    for rec in candidate['code']:
        if rec['path'].endswith('development_data.pt'):
            old=torch.load(rec['path'],weights_only=False)
            old_pairs.extend(p for s in old for pair in s['pairs'] for p in pair)
    assert not any(scene in Path(p).parts for p in old_pairs for scene in pre['scenes'])
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
    exact=load_exact(candidate['weights'][1]);risk=ScheduleRisk('mlp').cuda().eval().requires_grad_(False)
    risk.load_state_dict(torch.load(candidate['weights'][0]['path'],weights_only=False)['risk'],strict=True)
    out.mkdir();protocol={'role':pre['role'],'preregistration':prepared['preregistration'],'inputs':records,
        'source':file_record(__file__),'runtime':runtime_versions(),'gpu':torch.cuda.get_device_name(),
        'old_training_calibration_development_sequence_overlap':False,'threshold':pre['threshold'],
        'no_training_no_threshold_changes':True,'per_sequence_gate':'same numerical gate applied diagnostically to each sequence; no averaging failures away'}
    q.write(out/'protocol.json',protocol);results={};parity=[]
    for length in (32,256):
        manifest=root/f'manifest_L{length}.json';assert check_final_test(manifest)=='development_validation'
        samples=[]
        for src,ds in load_manifest(manifest):
            for i in range(len(ds)):
                rgb,gt,valid=ds[i];pairs=q.paths(ds,i)
                samples.append({'rgb':rgb,'gt':gt,'valid':valid.bool(),'source':src,'scene':q.scene(pairs[0]),'pairs':pairs})
        torch.save(samples,out/f'L{length}_data.pt');results[f'L{length}']={};dense=None
        for name,model in {'dense':exact,'output_k2':RGBFlowStream(exact),'mlp_q50':PreflowScheduledFlow(exact,risk,pre['threshold'])}.items():
            measured,preds=evaluate(model,samples)
            if name=='dense':baseline=measured;dense=preds
            measured['quality_gate']=quality_gate(baseline['metrics'],measured['metrics'])
            measured['speedup_vs_dense']=baseline['mean_ms']/measured['mean_ms']
            measured['sequence_gates']={}
            for scene in sorted({s['scene'] for s in samples}):
                a=q.aggregate([r for r in baseline['metrics']['clips'] if r['scene']==scene])
                b=q.aggregate([r for r in measured['metrics']['clips'] if r['scene']==scene])
                measured['sequence_gates'][scene]={'baseline':a['balanced'],'candidate':b['balanced'],'gate':quality_gate(a,b)}
            count=0
            if name!='dense':
                for ci,tele in enumerate(measured['telemetry']):
                    for t,info in enumerate(tele['frames']):
                        if info['full_refresh']:assert torch.equal(preds[ci][t],dense[ci][t]);count+=1
                        if info.get('reason')=='risk':assert info['flow_calls']==0
            parity.append({'length':length,'policy':name,'exact_refresh_comparisons':count})
            torch.save(preds,out/f'L{length}_{name}_predictions.pt')
            results[f'L{length}'][name]=measured;q.write(out/'results.json',results)
            print('RESULT',length,name,measured['metrics']['balanced'],measured['quality_gate'],'ms',measured['mean_ms'],flush=True)
    for r in records+[protocol['source']]:assert file_record(r['path'])==r
    q.write(out/'integrity.json',{'hashes_unchanged':True,'exact_refresh_comparisons':sum(p['exact_refresh_comparisons'] for p in parity),'parity':parity,'candidate_promoted':False})
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
