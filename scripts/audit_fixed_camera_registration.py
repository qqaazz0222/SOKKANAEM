"""Chance-normalized RGB/depth registration check with an unregistered control. No inference."""
import io
import json
import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from sokkanaem.protocol import file_record
from scripts.quality_refinement import write
from scripts.admit_fixed_camera_urfd import ROOT, SBM, ARCHIVE, SHIFTS, REGISTRATION_FRAMES, BORDER_PX

BINS = 32


def nmi(depth, gray, valid):
    """Normalized mutual information; 0 when either channel is constant over the overlap."""
    if valid.sum() < 1000: return 0.
    d, g = depth[valid], gray[valid]
    hist = np.histogram2d(d, g, bins=BINS, range=[[d.min(), d.max()+1e-6], [0, 256]])[0]
    p = hist/hist.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = lambda q: float(-np.sum(q[q > 0]*np.log(q[q > 0])))
        hx, hy = h(px), h(py)
        if hx <= 0 or hy <= 0: return 0.
        return (hx + hy - h(p))/np.sqrt(hx*hy)


def sweep(depth, gray):
    inner = np.zeros(depth.shape, bool)
    inner[BORDER_PX:-BORDER_PX, BORDER_PX:-BORDER_PX] = True
    rows = []
    for dy in SHIFTS:
        for dx in SHIFTS:
            moved = np.roll(np.roll(depth, dy, axis=0), dx, axis=1)
            rows.append({'dx': dx, 'dy': dy, 'nmi': nmi(moved, gray, inner & (moved > 0))})
    best = max(rows, key=lambda r: r['nmi'])
    return {'best': best, 'nmi_at_zero': next(r['nmi'] for r in rows if r['dx'] == r['dy'] == 0),
            'best_offset_px': float(np.hypot(best['dx'], best['dy']))}


def main():
    out = ROOT/'registration_audit.json'
    if out.exists(): raise FileExistsError('Immutable audit')
    with zipfile.ZipFile(ARCHIVE) as z:
        originals = {int(Path(n).stem.split('-')[-1]): np.array(Image.open(io.BytesIO(z.read(n))))
                     for n in z.namelist() if n.lower().endswith('.png')}
    rows = []
    for i in REGISTRATION_FRAMES:
        gray = cv2.imread(str(SBM/'original_sbm/input'/f'in{i:06d}.png'), cv2.IMREAD_GRAYSCALE)
        corrected = np.array(Image.open(SBM/'corrected_sbm/depth'/f'd{i:06d}.png'))
        rows.append({'id': i, 'corrected': sweep(corrected, gray),
                     'control_unregistered_original': sweep(originals[i], gray)})
    c6 = sum(r['corrected']['best_offset_px'] <= 1.5 for r in rows)
    c7 = sum(r['control_unregistered_original']['best_offset_px'] > 1.5 for r in rows)
    c6_pass, c7_pass = c6 >= 7, c7 >= 6
    prior = json.loads((ROOT/'urfd_admission.json').read_text())
    report = {'source': file_record(__file__), 'protocol': file_record(ROOT/'protocol_registration.json'),
              'prior_admission': file_record(ROOT/'urfd_admission.json'),
              'no_model_predictions': True, 'bins': BINS, 'frames_evaluated': len(rows),
              'C5_edge_test_failed_and_preserved': not prior['C5_pass'],
              'C6_frames_peaking_at_origin': c6, 'C6_pass': c6_pass,
              'C7_control_frames_peaking_off_origin': c7, 'C7_metric_discriminates': c7_pass,
              'C6_void': not c7_pass,
              'registration_corroborated': bool(c6_pass and c7_pass),
              'C3_pass': prior['C3_pass'], 'C4_pass': prior['C4_pass'], 'C1_pass': prior['C1_pass'],
              'admitted_as_masked_gt': bool(prior['C3_pass'] and prior['C4_pass'] and c6_pass and c7_pass),
              'scope': 'one event, one location; alignment to released RGB only, not extrinsic calibration',
              'sweeps': rows}
    write(out, report)
    print('C6', c6, c6_pass, 'C7', c7, c7_pass, 'ADMITTED', report['admitted_as_masked_gt'], flush=True)


if __name__ == '__main__': main()
