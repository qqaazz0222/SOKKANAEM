"""Data-only depth-range/invalid-boundary audit after failed initial admission."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scripts.fixed_camera_admission import complete_prefix_members, ur_depth_m
from scripts.acquire_fixed_camera_sbm import ROOT
from sokkanaem.protocol import file_record


def main():
    out=ROOT/'depth_diagnostic.json'
    if out.exists(): raise FileExistsError('Immutable diagnostic')
    rows=[]
    for name,a in complete_prefix_members('work_dirs/qfixed_public_probe_20260910/ur_fall01_cam1_depth.zip'):
        i=int(Path(name).stem.split('-')[-1])
        b=np.array(Image.open(ROOT/'corrected_sbm/depth'/f'd{i:06d}.png'))
        old=np.array(Image.open(ROOT/'original_sbm/depth'/f'd{i:06d}.png'))
        mm=ur_depth_m(a)*1000
        lower=float(mm[a>0].min()); upper=float(mm.max())
        unsupported=(b>0)&((b<lower-1)|(b>upper+1))
        radius=[]
        for r in (0,1,2,3,4,8):
            valid=cv2.erode((b>0).astype(np.uint8),np.ones((2*r+1,2*r+1),np.uint8),
                            borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
            radius.append({'radius':r,'valid_pixels':int(valid.sum()),
                           'outside_original_range_pixels':int((valid&unsupported).sum()),
                           'minimum_kept':int(b[valid].min())})
        rows.append({'id':i,'original_min_mm':lower,'original_max_mm':upper,
                     'corrected_min':int(b[b>0].min()),'archive_matches_correction':bool(np.array_equal(b,old)),
                     'radii':radius})
    result={'source':file_record(__file__),'initial_admission':file_record(ROOT/'admission.json'),
            'no_model_predictions':True,'diagnosis_not_filter_approval':True,'frames':rows}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
