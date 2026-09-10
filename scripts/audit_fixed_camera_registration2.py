"""Registration sweep on a shift-invariant pixel set, after two instrument failures. No inference."""
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
from scripts.audit_fixed_camera_registration import BINS, nmi

RADIUS = max(SHIFTS)


def sweep(depth, gray):
    """Fixed evaluation set: every pixel stays valid under every shift, so scores are comparable."""
    keep = cv2.erode((depth > 0).astype(np.uint8), np.ones((2*RADIUS+1, 2*RADIUS+1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    keep[:BORDER_PX] = keep[-BORDER_PX:] = False
    keep[:, :BORDER_PX] = keep[:, -BORDER_PX:] = False
    ys, xs = np.nonzero(keep)
    g = gray[ys, xs]
    rows = [{'dx': dx, 'dy': dy, 'nmi': nmi(depth[ys-dy, xs-dx], g, np.ones(len(ys), bool))}
            for dy in SHIFTS for dx in SHIFTS]
    best = max(rows, key=lambda r: r['nmi'])
    return {'evaluated_pixels': int(keep.sum()), 'best': best,
            'nmi_at_zero': next(r['nmi'] for r in rows if r['dx'] == r['dy'] == 0),
            'best_offset_px': float(np.hypot(best['dx'], best['dy'])),
            'boundary_peak': bool(abs(best['dx']) == RADIUS or abs(best['dy']) == RADIUS)}


def main():
    out = ROOT/'registration_audit2.json'
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
    boundary = [r['id'] for r in rows for k in ('corrected', 'control_unregistered_original') if r[k]['boundary_peak']]
    c8 = not boundary
    c9 = sum(r['control_unregistered_original']['best_offset_px'] > 1.5 for r in rows)
    c10 = sum(r['corrected']['best_offset_px'] <= 1.5 for r in rows)
    prior = json.loads((ROOT/'urfd_admission.json').read_text())
    prior2 = json.loads((ROOT/'registration_audit.json').read_text())
    corroborated = bool(c8 and c9 >= 6 and c10 >= 7)
    report = {'source': file_record(__file__), 'protocol': file_record(ROOT/'protocol_registration2.json'),
              'prior_admission': file_record(ROOT/'urfd_admission.json'),
              'prior_registration_attempt': file_record(ROOT/'registration_audit.json'),
              'no_model_predictions': True, 'bins': BINS, 'shift_radius_px': RADIUS,
              'attempt1_edge_test_failed': not prior['C5_pass'],
              'attempt2_mutual_information_failed': not prior2['C6_pass'],
              'C8_no_boundary_peaks': c8, 'C8_boundary_peak_frames': sorted(set(boundary)),
              'C9_control_frames_off_origin': c9, 'C9_pass': c9 >= 6,
              'C10_corrected_frames_at_origin': c10, 'C10_pass': c10 >= 7,
              'void': not c8, 'registration_corroborated': corroborated,
              'C1_pass': prior['C1_pass'], 'C3_pass': prior['C3_pass'], 'C4_pass': prior['C4_pass'],
              'admitted_as_masked_gt': bool(prior['C3_pass'] and prior['C4_pass'] and corroborated),
              'scope': 'one event, one location; alignment to released RGB only, not extrinsic calibration',
              'sweeps': rows}
    write(out, report)
    print('C8', c8, 'C9', c9, 'C10', c10, 'CORROBORATED', corroborated,
          'ADMITTED', report['admitted_as_masked_gt'], flush=True)


if __name__ == '__main__': main()
