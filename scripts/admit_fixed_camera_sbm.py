"""Data-only admission checks and a preregistered, single-event pilot contract."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from sokkanaem.protocol import file_record
from scripts.fixed_camera_admission import complete_prefix_members, ur_depth_m, strict_pairs, indexed_frames
from scripts.acquire_fixed_camera_sbm import ROOT


def background_motion(pairs, masks):
    """Sparse annotated-background image-motion diagnostic, not pose truth."""
    cv2.setRNGSeed(731)
    orb = cv2.ORB_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    annotated = [(i,r,d) for i,r,d in pairs if i in masks]
    rows = []
    for (i,r,_),(j,s,_) in zip(annotated, annotated[1:]):
        a,b = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in (r,s)]
        ma,mb = [cv2.erode((np.array(Image.open(masks[t]))==0).astype(np.uint8)*255,
                           np.ones((7,7),np.uint8)) for t in (i,j)]
        ka,da = orb.detectAndCompute(a,ma); kb,db = orb.detectAndCompute(b,mb)
        row = {'frame_ids':[i,j], 'scope':'annotated background only; sparse interval; no pose GT'}
        if da is None or db is None:
            rows.append(dict(row, classified=False)); continue
        good = [m for match in bf.knnMatch(da,db,k=2) if len(match)==2
                for m,n in [match] if m.distance < .7*n.distance]
        row['matches'] = len(good)
        if len(good) < 20:
            rows.append(dict(row, classified=False)); continue
        x = np.float32([ka[m.queryIdx].pt for m in good])
        y = np.float32([kb[m.trainIdx].pt for m in good])
        affine, keep = cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=2.)
        if affine is None:
            rows.append(dict(row, classified=False)); continue
        inside = keep[:,0].astype(bool)
        row.update(classified=True, inliers=int(inside.sum()),
                   median_inlier_displacement_px=float(np.median(np.linalg.norm(y[inside]-x[inside],axis=1))),
                   translation_px=float(np.linalg.norm(affine[:,2])),
                   rotation_deg=float(np.degrees(np.arctan2(affine[1,0],affine[0,0]))),
                   scale=float(np.linalg.norm(affine[:,0])))
        rows.append(row)
    return rows


def main():
    out=ROOT/'admission.json'
    if out.exists(): raise FileExistsError('Immutable admission')
    acquired=json.loads((ROOT/'acquired.json').read_text())
    for rec in acquired['frames']: assert file_record(rec['path'])==rec
    original=ROOT/'original_sbm'; corrected=ROOT/'corrected_sbm'
    pairs=strict_pairs(list((original/'input').glob('*.png')), list((corrected/'depth').glob('*.png')))
    assert len(pairs)==160 and pairs[0][0]==1 and pairs[-1][0]==160
    masks=indexed_frames(list((corrected/'groundtruth').glob('*.png')))
    counts=[]
    for i,r,d in pairs:
        rgb=np.array(Image.open(r)); depth=np.array(Image.open(d))
        assert rgb.shape==(480,640,3) and depth.shape==(480,640) and depth.dtype==np.uint16
        counts.append({'id':i,'valid_fraction':float(np.mean(depth>0)),
                       'saturated_65535':int(np.sum(depth==65535)), 'max':int(depth.max())})
    rgb_prefix=Path('work_dirs/qfixed_public_probe_20260910/ur_fall01_cam1_rgb.zip')
    depth_prefix=Path('work_dirs/qfixed_public_probe_20260910/ur_fall01_cam1_depth.zip')
    raw_rgb=complete_prefix_members(rgb_prefix); raw_depth=complete_prefix_members(depth_prefix)
    rgb_matches=[]
    for name,a in raw_rgb:
        i=int(Path(name).stem.split('-')[-1])
        rgb_matches.append({'id':i,'pixel_exact':bool(np.array_equal(a,np.array(Image.open(pairs[i-1][1]))))})
    # Independent source formula, with no fit to Q0, GT error, or affine scale.
    scale_samples=[]
    for name,a in raw_depth:
        i=int(Path(name).stem.split('-')[-1]); b=np.array(Image.open(pairs[i-1][2]))
        mm=ur_depth_m(a)*1000
        source_values=np.unique(np.rint(mm[a>0]).astype(int))
        target=b[b>0].astype(int); nearest=np.searchsorted(source_values,target)
        lo=source_values[np.clip(nearest-1,0,len(source_values)-1)]
        hi=source_values[np.clip(nearest,0,len(source_values)-1)]
        distance=np.minimum(abs(target-lo),abs(target-hi))
        scale_samples.append({'id':i,'original_mm_quantiles':np.quantile(mm[a>0],[.01,.5,.99]).tolist(),
                              'corrected_value_quantiles':np.quantile(target,[.01,.5,.99]).tolist(),
                              'within_1mm_of_original_rounded_value_fraction':float(np.mean(distance<=1))})
    assert rgb_matches and all(x['pixel_exact'] for x in rgb_matches)
    assert scale_samples
    # This corroborates documented rescaling, not a per-pixel registration proof.
    compatible=all(x['within_1mm_of_original_rounded_value_fraction']>=.99 for x in scale_samples)
    train_roots=[Path('configs'),Path('manifests'),Path('work_dirs/q0-calib-s0/config.toml'),
                 Path('work_dirs/qschedule_20260909/protocol.json'),Path('work_dirs/qpreflow_20260909/protocol.json')]
    searched=[]; matches=[]
    for root in train_roots:
        for p in sorted(root.rglob('*')) if root.is_dir() else [root]:
            if not p.is_file() or p.suffix not in ('.toml','.json','.yaml','.yml'): continue
            searched.append(file_record(p))
            if any(s in p.read_text().lower() for s in ('urfd','fall01cam1','fall-01-cam1','sbm-rgbd')):
                matches.append(str(p))
    report={'source':file_record(__file__),'helper':file_record('scripts/fixed_camera_admission.py'),
            'acquisition':file_record(ROOT/'acquired.json'),'pairs':pairs,'masks':masks,
            'frame_count':160,'pixel_checks':counts,'raw_rgb_prefix_matches':rgb_matches,
            'depth_scale_crosscheck':scale_samples,'scale_compatible':compatible,
            'source_prefixes':[file_record(rgb_prefix),file_record(depth_prefix)],
            'mount_evidence':'URFD official documentation: camera 1 ceiling mounted',
            'registration_evidence':'SBM official June 2017 update explicitly scales and registers fall01cam1 depth',
            'timing_evidence':'SBM synchronized RGB-D and complete identical frame-ID sets; no per-frame capture timestamps in derivative',
            'background_motion_diagnostic':background_motion(pairs,masks),
            'searched_training_records':searched,'prior_record_matches':matches,
            'pretraining_independence':'unknown','group':'URFD fall-01, cam0/cam1 and SBM derivatives are same event',
            'scope':'one-event pilot only; location diversity and long-horizon validation remain absent',
            'admitted_for_single_event_pilot':compatible and not matches,
            'no_predictions_no_training':True}
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('ADMISSION',report['admitted_for_single_event_pilot'],'RGB matches',len(rgb_matches),
          'scale samples',len(scale_samples),'scale compatible',compatible,
          'background',report['background_motion_diagnostic'],flush=True)


if __name__=='__main__': main()
