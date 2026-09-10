"""Bounded public-source format probe; no depth-model evaluation or training."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import zipfile
from urllib.request import urlopen

import numpy as np
from PIL import Image

from sokkanaem.protocol import file_record


ROOT = Path('work_dirs/qfixed_public_probe_20260910')
SOURCES = {
    'ur_fall01_cam1_depth.zip': 'https://fenix.ur.edu.pl/~mkepski/ds/data/fall-01-cam1-d.zip',
    'ur_fall01_cam1_rgb.zip': 'https://fenix.ur.edu.pl/~mkepski/ds/data/fall-01-cam1-rgb.zip',
    'sbm_fall01cam1_update.zip': 'https://rgbd2017.na.icar.cnr.it/SBM-RGBDdataset/Shadows/Updatefall01cam1.zip',
}


def fetch(item):
    name, url = item
    p = ROOT / name
    result = {'url': url, 'purpose': 'format inspection only; not a scored holdout'}
    try:
        with urlopen(url, timeout=30) as r:
            result['status'] = r.status
            result['declared_bytes'] = int(r.headers.get('Content-Length', 0))
            if result['declared_bytes'] > 150_000_000:
                raise ValueError('Probe download exceeds 150 MB cap')
            total = 0
            with p.open('xb') as f:
                while block := r.read(1024 * 1024):
                    total += len(block)
                    if total > 150_000_000:
                        raise ValueError('Probe transfer exceeds 150 MB cap')
                    f.write(block)
            if result['declared_bytes']:
                assert total == result['declared_bytes']
        result['archive'] = file_record(p)
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()
            assert bad is None, bad
            members = [n for n in z.namelist() if n.lower().endswith('.png')]
            result['png_count'] = len(members)
            groups = {}
            for member in members:
                groups.setdefault(str(Path(member).parent), []).append(member)
            result['groups'] = []
            for parent, files in sorted(groups.items()):
                files.sort()
                samples = []
                for i in sorted({0, len(files)//2, len(files)-1}):
                    a = np.array(Image.open(io.BytesIO(z.read(files[i]))))
                    samples.append({'member': files[i], 'shape': list(a.shape),
                                    'dtype': str(a.dtype), 'min': int(a.min()),
                                    'max': int(a.max()), 'nonzero_fraction': float(np.mean(a != 0))})
                result['groups'].append({'parent': parent, 'count': len(files), 'samples': samples})
        result['complete'] = True
    except Exception as e:
        result.update(complete=False, error=f'{type(e).__name__}: {e}')
    print(json.dumps(result), flush=True)
    return result


def main():
    ROOT.mkdir(exist_ok=False)
    protocol = {'sources': SOURCES, 'script': file_record(__file__),
                'no_model_predictions': True, 'no_training': True,
                'selection': 'first UR fall sequence, ceiling camera, and corresponding official SBM correction; chosen for format inspection, not quality'}
    (ROOT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, SOURCES.items()))
    (ROOT/'results.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
