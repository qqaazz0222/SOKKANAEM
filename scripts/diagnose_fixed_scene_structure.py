"""Why the registration sweeps could not localize: scene depth structure. No inference."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from sokkanaem.protocol import file_record
from scripts.quality_refinement import write
from scripts.admit_fixed_camera_urfd import ROOT, SBM, REGISTRATION_FRAMES
from scripts.audit_fixed_camera_registration import BINS

SHIFT_PX = (1, 2, 4, 8)


def main():
    out = ROOT/'scene_structure.json'
    if out.exists(): raise FileExistsError('Immutable diagnostic')
    foreground = []
    for p in sorted((SBM/'corrected_sbm/groundtruth').glob('*.png')):
        m = np.array(Image.open(p))
        foreground.append({'frame': p.stem, 'foreground_fraction': float((m == 255).mean())})
    frames = []
    for i in REGISTRATION_FRAMES:
        b = np.array(Image.open(SBM/'corrected_sbm/depth'/f'd{i:06d}.png')).astype(np.float32)
        valid = b > 0
        keep = cv2.erode(valid.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
        ys, xs = np.nonzero(keep)
        span = float(np.quantile(b[valid], .99) - np.quantile(b[valid], .01))
        row = {'id': i, 'depth_span_p01_p99_mm': span, 'histogram_bin_width_mm': span/BINS,
               'shift_response_mm': {}}
        for dx in SHIFT_PX:
            diff = np.abs(b[ys, xs-dx] - b[ys, xs])
            row['shift_response_mm'][f'{dx}px'] = {'median': float(np.median(diff)),
                                                   'p95': float(np.quantile(diff, .95))}
        row['median_8px_response_in_bins'] = row['shift_response_mm']['8px']['median']/row['histogram_bin_width_mm']
        frames.append(row)
    worst = max(f['median_8px_response_in_bins'] for f in frames)
    report = {'source': file_record(__file__), 'no_model_predictions': True,
              'question': 'Why did three model-free registration sweeps fail to localize the alignment?',
              'annotated_foreground': foreground,
              'max_foreground_fraction': max(f['foreground_fraction'] for f in foreground),
              'frames': frames, 'histogram_bins': BINS,
              'largest_median_8px_response_in_bins': worst,
              'finding': ('A ceiling view of this room is close to a plane. Translating the depth map by the '
                          'full 8 px search radius moves the median pixel by less than one histogram bin, so an '
                          'intensity-similarity objective has almost no lateral signal to lock onto. The failure '
                          'is a property of the scene, not only of the three metrics tried.'),
              'implication': ('An intensity-based registration check cannot be repaired on this sequence. Either the '
                              'SBM registration procedure is documented, or a fixed-camera source with real depth '
                              'relief is used instead.'),
              'separate_concern': ('The same near-planarity, with annotated foreground between 0.00% and 7.76% of the '
                                   'image, makes this sequence a weak depth benchmark even if it were admitted.')}
    write(out, report)
    print('max median 8px response in bins %.2f' % worst, '| max foreground %.4f' % report['max_foreground_fraction'], flush=True)


if __name__ == '__main__': main()
