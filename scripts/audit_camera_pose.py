"""Camera-motion audit, never infer fixed mounting from a sequence name."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record
from scripts.quality_refinement import write


def load_pose(path):
    rows=np.loadtxt(path,comments='#',ndmin=2)
    if rows.shape[1]!=8:raise ValueError('Expected timestamp tx ty tz qx qy qz qw')
    rows=rows[np.isfinite(rows).all(1)];norm=np.linalg.norm(rows[:,4:],axis=1)
    rows=rows[norm>1e-8];rows[:,4:]/=np.linalg.norm(rows[:,4:],axis=1)[:,None]
    return rows[np.argsort(rows[:,0],kind='stable')]


def summarize(poses,timestamps=None):
    if not len(poses):return {'coverage':0.,'label':'unclassified_missing_pose'}
    if timestamps is not None:
        ts=np.array(timestamps);idx=np.searchsorted(poses[:,0],ts)
        hi=np.clip(idx,0,len(poses)-1);lo=np.clip(idx-1,0,len(poses)-1)
        idx=np.where(abs(poses[hi,0]-ts)<abs(poses[lo,0]-ts),hi,lo)
        error=abs(poses[idx,0]-ts);ok=error<=.05;matched=poses[idx[ok]]
        coverage=float(ok.mean());match_max=float(error[ok].max()) if ok.any() else None
    else:matched=poses;coverage=1.;match_max=None
    if not len(matched):return {'coverage':coverage,'label':'unclassified_missing_pose'}
    distance=np.linalg.norm(matched[:,1:4]-matched[0,1:4],axis=1)
    cosine=np.clip(np.abs(matched[:,4:]@matched[0,4:]),0,1)
    angle=np.rad2deg(2*np.arccos(cosine));d95=float(np.quantile(distance,.95));a95=float(np.quantile(angle,.95))
    low=coverage>=.95 and d95<=.02 and a95<=1.
    return {'coverage':coverage,'matched_frames':len(matched),'max_matched_timestamp_error_s':match_max,
        'translation_p95_m':d95,'translation_max_m':float(distance.max()),
        'rotation_p95_deg':a95,'rotation_max_deg':float(angle.max()),
        'label':'low_motion_not_mount_verified' if low else ('camera_motion_present' if coverage>=.95 else 'unclassified_incomplete_pose'),
        'fixed_mount_verified':False}


def main():
    out=Path('work_dirs/qfixed_scope_20260910')
    if out.exists():raise FileExistsError('New immutable pose audit required')
    out.mkdir();scope=Path('paper/streaming_draft/FIXED_CAMERA_SCOPE.md')
    root=Path('work_dirs/qpreflow_20260909');parent=json.loads((root/'protocol.json').read_text())
    protocol={'scope':file_record(scope),'source':file_record(__file__),
        'rule':'pose nearest<=50ms; coverage>=95%; p95 displacement<=2cm and rotation<=1degree from first matched pose; diagnostic not proof of mount',
        'no_model_inference_or_retraining':True,'data':parent['development']}
    write(out/'protocol.json',protocol);result={'clips':[],'sequences':[]};records=[];seen={}
    for length,rec in zip(('L32','L256'),parent['development']):
        assert file_record(rec['path'])==rec
        data=torch.load(rec['path'],weights_only=False)
        for ci,s in enumerate(data):
            path=Path(s['pairs'][0][0]).parent.parent/'groundtruth.txt'
            if path not in seen:seen[path]=load_pose(path);records.append(file_record(path))
            timestamps=[float(Path(pair[0]).stem) for pair in s['pairs']]
            result['clips'].append({'length':length,'clip':ci,'scene':str(path.parent),'pose':str(path),**summarize(seen[path],timestamps)})
    for path,poses in seen.items():result['sequences'].append({'scene':str(path.parent),**summarize(poses)})
    result['pose_files']=records
    result['interpretation']='Mixed development remains mixed. Low pose motion does not confirm a tripod; motion/coverage results cannot exclude GT noise or short unobserved excursions.'
    for r in records+[protocol['source'],protocol['scope']]:assert file_record(r['path'])==r
    write(out/'audit.json',result)
    lines=['# 기존 개발 영상의 카메라 움직임 감사','',
        '2026-09-10. 깊이 모델을 재실행하거나 임계값을 변경하지 않고 pose 기록만 분석했다.',
        '첫 유효 pose 기준 변위. 저움직임 표시는 장착 고정의 증명이 아니다. 일부 구간에만 유효 pose가 있으면 그 부분만으로 고정을 판정하지 않는다.','',
        '| 구간 | 시퀀스 | 클립 | Pose 대응률 | 이동 p95(m) | 회전 p95(deg) | 판정 |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in result['clips']:
        lines.append(f'| {r["length"]} | {Path(r["scene"]).name} | {r["clip"]} | {100*r["coverage"]:.1f}% | {r.get("translation_p95_m",float("nan")):.4f} | {r.get("rotation_p95_deg",float("nan")):.3f} | {r["label"]} |')
    lines += ['','기존 4개 개발 시퀀스 전체를 고정 카메라 벤치마크로 재명명하지 않는다. 고정 설치가 확인된 별도 장소의 RGB-D 자료가 주 검증에 필요하다.',
        '`work_dirs/qfixed_scope_20260910/audit.json`에 전체 시퀀스 진단, 최대 변위/회전, 입력 hash를 보존했다.']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
