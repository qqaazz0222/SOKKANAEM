"""Screen local SBM category archives for depth relief and foreground share. No inference, no admission."""
import argparse
import io
import json
import zipfile
from pathlib import Path, PurePosixPath
import cv2
import numpy as np
from PIL import Image
from sokkanaem.protocol import file_record
from scripts.quality_refinement import write
from scripts.admit_fixed_camera_urfd import ROOT, SBM
from scripts.audit_fixed_camera_registration import BINS

SHIFT_PX = 8
STRIDE = 20


def relief(depth):
    """Median depth change under a full-search-radius shift, in histogram bins. 0.31 failed on fall01cam1."""
    valid = depth > 0
    keep = cv2.erode(valid.astype(np.uint8), np.ones((2*SHIFT_PX+1,)*2, np.uint8)).astype(bool)
    if keep.sum() < 1000: return None
    ys, xs = np.nonzero(keep)
    span = float(np.quantile(depth[valid], .99) - np.quantile(depth[valid], .01))
    if span <= 0: return None
    d = np.abs(depth[ys, xs-SHIFT_PX].astype(np.float64) - depth[ys, xs].astype(np.float64))
    return {'depth_span_p01_p99': span, 'bin_width': span/BINS,
            'median_shift_response': float(np.median(d)),
            'median_shift_response_in_bins': float(np.median(d)/(span/BINS))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, default=SBM/'Shadows.zip')
    ap.add_argument('--out', type=Path, default=ROOT/'sbm_relief_screen.json')
    args = ap.parse_args()
    out = args.out
    if out.exists(): raise FileExistsError('Immutable screen')
    groups = {}
    with zipfile.ZipFile(args.archive) as z:
        for n in z.namelist():
            if not n.endswith('.png') or n.startswith('__MACOSX'): continue
            parts = PurePosixPath(n).parts
            if len(parts) < 4: continue
            groups.setdefault(parts[1], {}).setdefault(parts[2], []).append(n)
        rows = []
        for name, kinds in sorted(groups.items()):
            depths = sorted(kinds.get('depth', []))
            reliefs = [r for n in depths[::STRIDE] for r in [relief(np.array(Image.open(io.BytesIO(z.read(n)))))] if r]
            shares = [float((np.array(Image.open(io.BytesIO(z.read(n)))) == 255).mean())
                      for n in sorted(kinds.get('groundtruth', []))]
            rows.append({'sequence': name, 'depth_frames': len(depths), 'annotated_frames': len(shares),
                         'sampled_frames': len(reliefs),
                         'median_relief_in_bins': float(np.median([r['median_shift_response_in_bins'] for r in reliefs])) if reliefs else None,
                         'min_relief_in_bins': float(min(r['median_shift_response_in_bins'] for r in reliefs)) if reliefs else None,
                         'median_depth_span': float(np.median([r['depth_span_p01_p99'] for r in reliefs])) if reliefs else None,
                         'median_foreground_share': float(np.median(shares)) if shares else None,
                         'max_foreground_share': max(shares) if shares else None})
    report = {'source': file_record(__file__), 'archive': file_record(args.archive),
              'no_model_predictions': True, 'no_admission_claim': True,
              'statistic': (f'Median absolute depth change under a {SHIFT_PX} px horizontal shift, divided by the '
                            f'width of one of {BINS} histogram bins spanning the frame p01 to p99 depth. Every '
                            f'{STRIDE}th depth frame is sampled.'),
              'reference': 'fall01cam1 scores 0.31 bins and three registration sweeps failed to localize on it.',
              'caveat': ('Relief and foreground share say whether a sequence could discriminate. They say nothing '
                         'about metric-depth provenance or registration documentation, which remain the blockers. '
                         'The depth encoding of these other sequences is undocumented, so none of them is admitted '
                         'by this screen.'),
              'sequences': rows}
    write(out, report)
    for r in rows:
        print(f"{r['sequence']:12s} relief {r['median_relief_in_bins']:.2f} bins  span {r['median_depth_span']:.0f}"
              f"  foreground {r['median_foreground_share']*100 if r['median_foreground_share'] is not None else float('nan'):.2f}%"
              f"  annotated {r['annotated_frames']}", flush=True)


if __name__ == '__main__': main()
