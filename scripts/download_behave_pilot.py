"""Acquire complete, selected BEHAVE TAR members, not the 55 GB archives.

Only run after licensed non-commercial use is confirmed. Acquisition selection
uses names/camera ID, never predictions. Partial archives are not called complete.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
from urllib.request import Request, urlopen
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record
from sokkanaem.behave_io import associate_timestamps

BASE = 'https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/'


def metadata(url):
    with urlopen(Request(url, method='HEAD'), timeout=30) as r:
        return {'url': url, 'bytes': int(r.headers['Content-Length']), 'etag': r.headers['ETag']}


def request_range(meta, start, end):
    req = Request(meta['url'], headers={'Range': f'bytes={start}-{end}', 'If-Match': meta['etag'],
                                       'Accept-Encoding': 'identity'})
    r = urlopen(req, timeout=30)
    expected = f'bytes {start}-{end}/{meta["bytes"]}'
    if r.status != 206 or r.headers.get('Content-Range') != expected or r.headers.get('ETag') != meta['etag']:
        r.close(); raise ValueError('Server did not honor exact immutable byte range')
    return r


class RemoteTar(io.RawIOBase):
    def __init__(self, meta):
        self.meta = meta; self.pos = 0; self.transferred = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.pos+offset if whence == 1 else self.meta['bytes']+offset
        if target < 0: raise ValueError('Negative seek')
        self.pos = target; return target
    def read(self, size=-1):
        if size < 0 or size > 2_000_000: raise ValueError('Header read cap')
        size = min(size, self.meta['bytes']-self.pos)
        if size <= 0: return b''
        with request_range(self.meta, self.pos, self.pos+size-1) as r:
            b = r.read(size+1)
        if len(b) != size: raise ValueError('Truncated archive header')
        self.pos += size; self.transferred += size
        return b


def index_selected(meta, wanted):
    source = RemoteTar(meta); selected = {}; members = []
    with tarfile.open(fileobj=source, mode='r:') as tar:
        for m in tar:
            p = PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts: raise ValueError('Unsafe member path')
            members.append({'name': m.name, 'bytes': m.size, 'offset': m.offset_data, 'regular': m.isfile()})
            if p.name in wanted:
                if not m.isfile() or p.name in selected: raise ValueError('Invalid/duplicate selected member')
                if m.offset_data+m.size > meta['bytes']: raise ValueError('Member extends beyond archive')
                selected[p.name] = members[-1]
            if set(selected) == set(wanted): break
    if set(selected) != set(wanted): raise ValueError(f'Missing members {set(wanted)-set(selected)}')
    return {'archive': meta, 'selected': selected, 'visited_headers': members,
            'header_bytes_transferred': source.transferred, 'whole_archive_acquired': False,
            'whole_archive_indexed': False}


def download_range(meta, start, size, path):
    if not 0 < size <= 2_000_000_000: raise ValueError('Per-file acquisition cap')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    partial = path.with_suffix(path.suffix+'.part')
    # Retries only resume the file created by this invocation, never arbitrary old partials.
    with partial.open('xb'): pass
    for attempt in range(4):
        have = partial.stat().st_size
        if have == size: break
        try:
            with request_range(meta, start+have, start+size-1) as r, partial.open('ab') as out:
                while have < size:
                    b = r.read(min(1024*1024, size-have))
                    if not b: raise IOError('Truncated member payload')
                    out.write(b); have += len(b)
            break
        except Exception:
            if attempt == 3: raise
            print('retry', path.name, partial.stat().st_size, flush=True)
    if partial.stat().st_size != size: raise ValueError('Incomplete payload')
    partial.rename(path)
    record = file_record(path)
    print('downloaded', path.name, size, flush=True)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--confirmed-noncommercial-research', action='store_true')
    ap.add_argument('--out', type=Path, default=Path('work_dirs/qbehave_pilot_acquisition_20260910'))
    args = ap.parse_args()
    if not args.confirmed_noncommercial_research: ap.error('User confirmation required')
    args.out.mkdir(parents=True, exist_ok=False)
    time_root = Path('work_dirs/qbehave_timestamps_20260910')
    inv = json.loads((time_root/'inventory.json').read_text())
    names = sorted(r['member'] for r in inv['timestamp_files'] if r['member'].endswith('.1.time.json'))
    selected = [n for n in names if '_Sub' in n][:2] + [n for n in names if '_empty_' in n][:1]
    if len(selected) != 3: raise ValueError('Expected two interaction sequences and one empty reference')
    prefixes = [n.removesuffix('.time.json') for n in selected]
    protocol = {'source': file_record(__file__), 'timestamps': file_record(time_root/'inventory.json'),
                'candidate': file_record('paper/streaming_draft/frozen_candidate.json'),
                'user_confirmation': '비상업적 연구야. 다운로드 진행해줘.',
                'selection': 'Date02 camera 1; lexicographically first two subject sequences and first empty reference',
                'selected_timestamps': selected, 'no_model_predictions': True,
                'purpose': 'acquisition/format pilot, not independent-location accuracy validation',
                'gt_admitted': False, 'max_selected_file_bytes': 2_000_000_000,
                'license': 'https://virtualhumans.mpi-inf.mpg.de/behave/license.html'}
    def write(name, data):
        (args.out/name).write_text(json.dumps(data, indent=2)+'\n')
    write('protocol.json', protocol)
    with tarfile.open(time_root/'date02_time.tar', 'r:') as tar:
        pairs = []
        for name in selected:
            blob = tar.extractfile(name).read()
            p = args.out/'times'/name; p.parent.mkdir(exist_ok=True); p.write_bytes(blob)
            ts = json.loads(blob)
            # 5 ms is a diagnostic pairing tolerance, not an admission threshold.
            a = associate_timestamps(ts['color'], ts['depth'], 5000)
            write(name+'.association.json', a)
            pairs.append({'name': name, 'color_count': a['color_count'], 'matched': a['matched'],
                          'coverage': a['coverage'], 'max_rgb_gap_us': a['color_interval_us']['max']})
    write('pairing_diagnostics.json', pairs)
    def acquire_kind(kind):
        suffix = 'color.mp4' if kind == 'color' else 'depth-reg.mp4'
        meta = metadata(BASE+f'video/date02_{kind}.tar')
        index = index_selected(meta, [p+'.'+suffix for p in prefixes])
        write(kind+'_index.json', index)
        results = []
        for name, m in index['selected'].items():
            results.append(download_range(meta, m['offset'], m['bytes'], args.out/'videos'/name))
        write(kind+'_downloaded.json', results)
        return results
    def calibrations():
        meta = metadata(BASE+'calibs.zip')
        if meta['bytes'] > 500_000_000: raise ValueError('Calibration cap')
        write('calibration_source.json', meta)
        path = args.out/'calibs.zip'
        record = download_range(meta, 0, meta['bytes'], path)
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad is not None: raise ValueError(f'Calibration CRC failure: {bad}')
            entries = [{'name': m.filename, 'bytes': m.file_size} for m in z.infolist()]
        write('calibration_inventory.json', {'archive': record, 'crc_verified': True, 'members': entries})
        return record
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(acquire_kind, kind) for kind in ('color', 'depth')]
        cal = pool.submit(calibrations)
        video_records = [r for f in futures for r in f.result()]
        calibration = cal.result()
    write('downloaded.json', {'protocol': file_record(args.out/'protocol.json'), 'videos': video_records,
          'calibration': calibration, 'all_selected_members_complete': True,
          'whole_video_archives_acquired': False, 'gt_admitted': False,
          'hashes_are_local_not_publisher_checksums': True})
    print('COMPLETE', args.out, flush=True)


if __name__ == '__main__': main()
