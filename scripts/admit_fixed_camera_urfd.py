"""Second data admission: complete URFD original depth as the reference. No inference."""
import io
import json
import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from sokkanaem.protocol import file_record
from scripts.fixed_camera_admission import ur_depth_m
from scripts.quality_refinement import write

ROOT = Path('work_dirs/qfixed_urfd_20260910')
SBM = Path('work_dirs/qfixed_validation_20260910')
ARCHIVE = ROOT/'fall-01-cam1-d.zip'
QUANTILES = (.05, .25, .50, .75, .95)
SHIFTS = range(-8, 9)
REGISTRATION_FRAMES = (1, 21, 41, 61, 81, 101, 121, 141)
# Fixed in this script, not tuned: rank-based so the edge count is identical across shifts.
EDGE_FRACTION = .02
EDGE_TOLERANCE_PX = 2
BORDER_PX = 8


def edge_score(depth_edges, rgb_distance):
    """Fraction of depth-discontinuity pixels within EDGE_TOLERANCE_PX of an RGB edge, per shift."""
    inner = np.zeros_like(depth_edges)
    inner[BORDER_PX:-BORDER_PX, BORDER_PX:-BORDER_PX] = True
    edges = depth_edges & inner
    total = int(edges.sum())
    best, rows = None, []
    for dy in SHIFTS:
        for dx in SHIFTS:
            moved = np.roll(np.roll(edges, dy, axis=0), dx, axis=1)
            hit = float(np.mean(rgb_distance[moved] <= EDGE_TOLERANCE_PX)) if total else 0.
            rows.append({'dx': dx, 'dy': dy, 'score': hit})
            if best is None or hit > best['score']: best = rows[-1]
    return {'edge_pixels': total, 'best': best, 'score_at_zero': next(r['score'] for r in rows if r['dx'] == r['dy'] == 0)}


def discontinuities(depth, valid):
    """Top EDGE_FRACTION of valid pixels by morphological depth gradient."""
    d = np.where(valid, depth.astype(np.float32), np.nan)
    filled = np.where(valid, depth.astype(np.float32), 0.)
    grad = cv2.morphologyEx(filled, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    solid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    g = grad[solid]
    if g.size == 0: return np.zeros_like(valid)
    del d
    return solid & (grad >= np.quantile(g, 1 - EDGE_FRACTION))


def main():
    out = ROOT/'urfd_admission.json'
    if out.exists(): raise FileExistsError('Immutable admission')
    protocol = file_record(ROOT/'protocol.json')
    assert ARCHIVE.stat().st_size == 42997797
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        originals = {}
        for name in z.namelist():
            if not name.lower().endswith('.png'): continue
            i = int(Path(name).stem.split('-')[-1])
            originals[i] = np.array(Image.open(io.BytesIO(z.read(name))))  # read() verifies CRC
    assert sorted(originals) == list(range(1, 161)), sorted(originals)[:5]

    frames, retentions, worst_quantile = [], [], []
    for i in range(1, 161):
        a = originals[i]
        b = np.array(Image.open(SBM/'corrected_sbm/depth'/f'd{i:06d}.png')).astype(np.int64)
        assert a.shape == b.shape == (480, 640)
        mm = ur_depth_m(a)*1000
        source = mm[a > 0]
        lo, hi = float(source.min()), float(mm.max())
        valid = b > 0
        target = b[valid]
        ladder = np.unique(np.rint(source).astype(int))
        near = np.searchsorted(ladder, target)
        low = ladder[np.clip(near-1, 0, len(ladder)-1)]
        high = ladder[np.clip(near, 0, len(ladder)-1)]
        within = float(np.mean(np.minimum(abs(target-low), abs(target-high)) <= 1))
        mask = valid & (b >= lo-1) & (b <= hi+1) & cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        retention = float(mask.sum()/valid.sum())
        deltas = [abs(float(np.quantile(b[mask], q)) - float(np.quantile(source, q))) for q in QUANTILES]
        retentions.append(retention); worst_quantile.append(max(deltas))
        frames.append({'id': i, 'original_min_mm': lo, 'original_max_mm': hi,
                       'C1_within_1mm_fraction': within,
                       'C2_below_original_min': int((valid & (b < lo-1)).sum()),
                       'C2_above_original_max': int((valid & (b > hi+1)).sum()),
                       'C3_valid_pixels': int(valid.sum()), 'C3_masked_pixels': int(mask.sum()),
                       'C3_retention': retention,
                       'C4_quantile_abs_diff_mm': dict(zip([f'p{int(q*100):02d}' for q in QUANTILES], deltas))})

    registration = []
    for i in REGISTRATION_FRAMES:
        rgb = cv2.imread(str(SBM/'original_sbm/input'/f'in{i:06d}.png'), cv2.IMREAD_GRAYSCALE)
        canny = cv2.Canny(rgb, 60, 160)
        distance = cv2.distanceTransform((canny == 0).astype(np.uint8), cv2.DIST_L2, 3)
        b = np.array(Image.open(SBM/'corrected_sbm/depth'/f'd{i:06d}.png'))
        a = originals[i]
        registration.append({'id': i,
                             'corrected': edge_score(discontinuities(b, b > 0), distance),
                             'control_unregistered_original': edge_score(discontinuities(a, a > 0), distance)})

    c1 = all(f['C1_within_1mm_fraction'] >= .99 for f in frames)
    c3 = float(np.median(retentions)) >= .90 and min(retentions) >= .80
    c4 = float(np.median(worst_quantile)) <= 50.
    c5 = all(r['corrected']['best']['dx'] == 0 and r['corrected']['best']['dy'] == 0
             and r['corrected']['score_at_zero'] > r['control_unregistered_original']['score_at_zero']
             for r in registration)
    report = {'source': file_record(__file__), 'protocol': protocol,
              'archive': file_record(ARCHIVE), 'archive_crc_verified': True,
              'reference_frames': 160, 'no_model_predictions': True,
              'C1_pass': c1, 'C1_min_fraction': min(f['C1_within_1mm_fraction'] for f in frames),
              'C1_max_fraction': max(f['C1_within_1mm_fraction'] for f in frames),
              'C3_pass': c3, 'C3_median_retention': float(np.median(retentions)), 'C3_min_retention': min(retentions),
              'C4_pass': c4, 'C4_median_worst_quantile_mm': float(np.median(worst_quantile)),
              'C4_max_worst_quantile_mm': max(worst_quantile),
              'C5_pass': c5, 'registration': registration,
              'edge_definition': {'fraction_of_valid_pixels': EDGE_FRACTION, 'tolerance_px': EDGE_TOLERANCE_PX,
                                  'border_excluded_px': BORDER_PX,
                                  'note': 'Depth edges are computed once and translated, so the edge set is identical at every shift.'},
              'admitted_as_masked_gt': bool(c3 and c4 and c5),
              'scope': 'one event, one location; no long-horizon or target-device validation',
              'frames': frames}
    write(out, report)
    print('C1', c1, 'C3', c3, 'C4', c4, 'C5', c5, 'ADMITTED', report['admitted_as_masked_gt'], flush=True)


if __name__ == '__main__': main()
