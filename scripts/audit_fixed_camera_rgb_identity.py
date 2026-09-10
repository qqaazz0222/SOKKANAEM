"""Are all 160 SBM input frames pixel-identical to the URFD originals? No inference."""
import io
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from sokkanaem.protocol import file_record
from scripts.quality_refinement import write
from scripts.admit_fixed_camera_urfd import ROOT, SBM

ARCHIVE = ROOT/'fall-01-cam1-rgb.zip'


def main():
    out = ROOT/'rgb_identity.json'
    if out.exists(): raise FileExistsError('Immutable audit')
    assert ARCHIVE.stat().st_size == 60161135
    rows = []
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        names = {int(Path(n).stem.split('-')[-1]): n for n in z.namelist() if n.lower().endswith('.png')}
        assert sorted(names) == list(range(1, 161)), sorted(names)[:5]
        for i in range(1, 161):
            a = np.array(Image.open(io.BytesIO(z.read(names[i]))))  # read() verifies CRC
            b = np.array(Image.open(SBM/'original_sbm/input'/f'in{i:06d}.png'))
            rows.append({'id': i, 'shape_match': a.shape == b.shape,
                         'pixel_exact': bool(a.shape == b.shape and np.array_equal(a, b))})
    exact = sum(r['pixel_exact'] for r in rows)
    report = {'source': file_record(__file__), 'archive': file_record(ARCHIVE), 'archive_crc_verified': True,
              'no_model_predictions': True, 'frames': 160, 'pixel_exact_frames': exact,
              'all_pixel_exact': exact == 160,
              'supersedes': 'The first admission confirmed 5 frames from an incomplete archive prefix.',
              'establishes': 'The released RGB of this sequence is the URFD original, unmodified, on every frame.',
              'does_not_establish': ('Nothing about the depth channel. Registration of depth to this RGB remains '
                                     'uncorroborated and the sequence remains unadmitted as ground truth.'),
              'per_frame': rows}
    write(out, report)
    print('pixel-exact', exact, '/ 160', flush=True)


if __name__ == '__main__': main()
