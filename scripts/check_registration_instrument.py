"""Known-shift calibration of an NMI instrument, NOT RGB-depth admission.

New experiment: historical admission scripts/results are never rewritten.
Coordinates, histogram edges and sample counts stay fixed over a sweep.
No model is loaded, no sensor reference is promoted by this diagnostic.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sokkanaem.protocol import file_record


def translated(a, dy, dx):
    """Integer translation with NaN padding; never wrap across an image border."""
    out = np.full(a.shape, np.nan, dtype=float)
    h, w = a.shape
    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)
    out[y0:y1, x0:x1] = a[y0-dy:y1-dy, x0-dx:x1-dx]
    return out


def sweep(reference, moving, radius=8, bins=32, stride=4):
    if reference.shape != moving.shape or reference.ndim != 2:
        raise ValueError('Matched two-dimensional inputs required')
    valid = np.isfinite(reference) & np.isfinite(moving)
    support = cv2.erode(valid.astype(np.uint8), np.ones((2*radius+1,)*2, np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    grid = np.zeros_like(support); grid[::stride, ::stride] = True
    ys, xs = np.nonzero(support & grid)
    if len(ys) < 32:
        raise ValueError('Insufficient common support')
    def edges(a):
        v = a[np.isfinite(a)]
        lo, hi = float(v.min()), float(v.max())
        return np.linspace(lo, hi if hi > lo else lo+1, bins+1)
    re, me = edges(reference), edges(moving)
    ref = reference[ys, xs]
    scores = []
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            moved = moving[ys+dy, xs+dx]
            assert np.isfinite(moved).all()
            hist = np.histogram2d(ref, moved, bins=(re, me))[0]
            assert int(hist.sum()) == len(ys)
            p = hist / hist.sum()
            def entropy(v):
                v = v[v > 0]
                return float(-(v*np.log(v)).sum())
            hr, hm, hj = entropy(p.sum(1)), entropy(p.sum(0)), entropy(p)
            score = (hr+hm-hj)/max(np.sqrt(hr*hm), 1e-15)
            scores.append({'dy': dy, 'dx': dx, 'score': float(score)})
    ordered = sorted(scores, key=lambda v: -v['score'])
    best = ordered[0]
    ties = [v for v in ordered if abs(v['score']-best['score']) <= 1e-10]
    return {'best': best, 'unique_peak': len(ties) == 1, 'tie_count': len(ties),
            'gap_to_second': best['score']-ordered[1]['score'],
            'sample_count': len(ys), 'histogram_edges': [re.tolist(), me.tolist()],
            'scores': scores}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('work_dirs/qregistration_instrument_20260910'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    shifts = [(0, 0), (3, -2), (-4, 4)]
    paths = [Path('work_dirs/qfixed_validation_20260910/corrected_sbm/depth')/f'd{i:06d}.png'
             for i in range(1, 142, 20)]
    protocol = {'purpose': 'same-modality known-shift instrument calibration only',
                'source': file_record(__file__), 'inputs': [file_record(p) for p in paths],
                'shifts_dy_dx': shifts, 'radius': 8, 'bins': 32, 'stride': 4,
                'success_rule': 'unique peak exactly at injected shift; flat control must be ambiguous',
                'can_admit_rgb_depth': False, 'no_model_inference': True,
                'fixed_candidate': file_record('paper/streaming_draft/frozen_candidate.json')}
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rng = np.random.default_rng(709)
    cases = [('synthetic_textured', rng.normal(size=(96, 128)))]
    for p in paths:
        a = np.array(Image.open(p), dtype=float); a[a <= 0] = np.nan
        cases.append((p.stem, a))
    rows = []
    for name, a in cases:
        for dy, dx in shifts:
            r = sweep(a, translated(a, dy, dx))
            r.update(case=name, injected=[dy, dx], recovered=r['unique_peak'] and
                     [r['best']['dy'], r['best']['dx']] == [dy, dx])
            rows.append(r)
        print(name, [r['recovered'] for r in rows[-3:]], flush=True)
    flat = sweep(np.ones((96, 128)), np.ones((96, 128)))
    result = {'protocol': file_record(args.out/'protocol.json'), 'cases': rows,
              'recovered': sum(r['recovered'] for r in rows), 'total': len(rows),
              'flat_control_ambiguous': not flat['unique_peak'], 'flat_control': flat,
              'actual_cross_modal_registration_verified': False, 'gt_admitted': False}
    assert all(file_record(r['path']) == r for r in protocol['inputs'])
    assert file_record(__file__) == protocol['source']
    (args.out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print({k: v for k, v in result.items() if k not in ('cases', 'flat_control')}, flush=True)


if __name__ == '__main__':
    main()
