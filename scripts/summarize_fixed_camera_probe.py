"""Snapshot completed and incomplete probe archives without resuming downloads."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
from sokkanaem.protocol import file_record
from scripts.probe_fixed_camera_data import ROOT, SOURCES


def main():
    records = []
    for name, url in SOURCES.items():
        path = ROOT / name
        row = {'url': url, 'file': file_record(path) if path.exists() else None,
               'complete_zip': path.exists() and zipfile.is_zipfile(path)}
        if row['complete_zip']:
            with zipfile.ZipFile(path) as z:
                assert z.testzip() is None
                groups = {}
                for member in z.namelist():
                    if member.lower().endswith('.png'):
                        groups.setdefault(str(Path(member).parent), []).append(member)
                row['groups'] = []
                for parent, members in sorted(groups.items()):
                    members.sort()
                    samples = []
                    for i in sorted({0, len(members)//2, len(members)-1}):
                        a = np.array(Image.open(io.BytesIO(z.read(members[i]))))
                        samples.append({'member': members[i], 'shape': list(a.shape),
                                        'dtype': str(a.dtype), 'min': int(a.min()),
                                        'max': int(a.max()), 'nonzero_fraction': float(np.mean(a != 0))})
                    row['groups'].append({'parent': parent, 'count': len(members), 'samples': samples})
        else:
            row['status'] = 'incomplete transfer retained; never use as dataset'
        records.append(row)
    snapshot = {'source': file_record(__file__), 'original_protocol': file_record(ROOT/'protocol.json'),
                'download_process_stopped': True,
                'reason': 'Bounded source/format search; slow URFD transfers stopped after obtaining complete SBM correction. No background download promise.',
                'no_model_predictions': True, 'archives': records}
    out = ROOT/'probe_snapshot.json'
    with out.open('x') as f:
        json.dump(snapshot, f, indent=2)
        f.write('\n')
    print(json.dumps(snapshot), flush=True)


if __name__ == '__main__':
    main()
