"""Small metadata-only BEHAVE acquisition after confirmed licensed research use.

Does not fetch RGB/depth/calibration, choose quality-favorable clips or admit GT.
"""
import argparse
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
from urllib.request import Request, urlopen
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.behave_io import checked_times
from sokkanaem.protocol import file_record

URL = 'https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/video/date02_time.tar'
MAX_BYTES = 10_000_000


def inspect_archive(blob):
    """Inspect regular JSON entries without extracting archive-controlled paths."""
    rows, seen = [], set()
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:') as tar:
        for member in tar:
            if member.isdir(): continue
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not member.isfile():
                raise ValueError(f'Unsafe archive member: {member.name}')
            if member.name in seen: raise ValueError('Duplicate archive member')
            seen.add(member.name)
            if not member.name.endswith('.json'): continue
            if member.size > 2_000_000: raise ValueError('JSON member size cap')
            with tar.extractfile(member) as stream:
                data = json.load(stream)
            row = {'member': member.name, 'bytes': member.size}
            try:
                for key in ('color', 'depth'):
                    t = checked_times(data[key], key)
                    dt = np.diff(t)
                    row[key] = {'count': len(t), 'first_us': float(t[0]), 'last_us': float(t[-1]),
                                'duration_s': float((t[-1]-t[0])/1e6),
                                'interval_us': dict(zip(('min', 'p50', 'p95', 'max'),
                                                       map(float, np.quantile(dt, [0, .5, .95, 1]))))}
                row['ordered_timestamps'] = True
            except (ValueError, KeyError, TypeError) as exc:
                row.update(ordered_timestamps=False, error=str(exc))
            rows.append(row)
    if not rows: raise ValueError('No timestamp JSON members')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--confirmed-noncommercial-research', action='store_true',
                    help='Use only after the user confirms non-commercial scientific research under BEHAVE terms')
    ap.add_argument('--out', type=Path, default=Path('work_dirs/qbehave_timestamps_20260910'))
    args = ap.parse_args()
    if not args.confirmed_noncommercial_research:
        ap.error('Explicit non-commercial research confirmation is required before dataset GET')
    if args.out.exists(): raise FileExistsError(args.out)
    with urlopen(Request(URL, method='HEAD'), timeout=25) as response:
        expected = int(response.headers['Content-Length'])
        etag = response.headers.get('ETag')
    if not 0 < expected <= MAX_BYTES: raise ValueError('Metadata download exceeds cap')
    args.out.mkdir(parents=True)
    protocol = {'url': URL, 'expected_bytes': expected, 'expected_etag': etag,
                'max_bytes': MAX_BYTES, 'source': file_record(__file__),
                'helper': file_record('sokkanaem/behave_io.py'),
                'candidate': file_record('paper/streaming_draft/frozen_candidate.json'),
                'license_url': 'https://virtualhumans.mpi-inf.mpg.de/behave/license.html',
                'confirmed_noncommercial_research_flag': True, 'no_model_inference': True,
                'purpose': 'timestamp inventory before selecting fixed-location validation clips'}
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    headers = {'If-Match': etag} if etag else {}
    with urlopen(Request(URL, headers=headers), timeout=25) as response:
        if response.status != 200: raise ValueError('Expected full metadata archive')
        if etag and response.headers.get('ETag') != etag: raise ValueError('Archive changed after HEAD')
        blob = response.read(MAX_BYTES+1)
    if len(blob) != expected: raise ValueError('Metadata archive byte count mismatch')
    archive = args.out/'date02_time.tar'; archive.write_bytes(blob)
    rows = inspect_archive(blob)
    result = {'protocol': file_record(args.out/'protocol.json'), 'archive': file_record(archive),
              'timestamp_files': rows, 'local_sha256_only_not_publisher_checksum': True,
              'rgb_or_depth_video_acquired': False, 'gt_admitted': False,
              'physical_location_groups_verified': False, 'selected_validation_clips': []}
    (args.out/'inventory.json').write_text(json.dumps(result, indent=2)+'\n')
    print({'timestamp_files': len(rows), 'ordered': sum(r['ordered_timestamps'] for r in rows),
           'gt_admitted': False}, flush=True)


if __name__ == '__main__': main()
