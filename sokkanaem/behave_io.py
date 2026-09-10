"""BEHAVE input checks; not a dataset-admission or model-quality decision.

Wire-format reference: vguzov/videoio, videoio/video_uint16.py (Apache-2.0).
BEHAVE reader: xiexh20/behave-dataset, data/video_reader.py.
This independent decoder intentionally starts at frame zero and never uses an
fps-based seek or a scaling filter on the packed depth channels.
"""
import json
from pathlib import Path
import subprocess

import numpy as np


def unpack_depth_yuv444(planes):
    """Decode videoio's parity-folded low byte and high byte, without RGB conversion."""
    a = np.asarray(planes)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[0] != 3:
        raise ValueError('Expected uint8 planes of shape (3, height, width)')
    high = a[2].astype(np.uint16)
    low = a[0].astype(np.uint16)
    low = np.where((high & 1) != 0, 255-low, low)
    return ((high << 8) | low).astype(np.uint16)


def depth_metres(depth):
    a = np.asarray(depth)
    if a.dtype != np.uint16 or a.ndim != 2:
        raise ValueError('Expected unscaled uint16 depth image')
    # Preserve invalid zero values: downstream validity is explicitly depth > 0.
    return a.astype(np.float32)/1000.0


def iter_depth_video(path, expected_count=None):
    """Decode all source frames exactly once; consume fully to verify count/exit.

    A clean EOF is not proof of a complete downloaded file. Acquisition needs its
    own byte/hash checks. The timestamp count provides an additional check here.
    """
    path = Path(path).resolve(strict=True)
    meta = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                           '-show_entries', 'stream=width,height,pix_fmt', '-of', 'json', str(path)],
                          check=True, capture_output=True, text=True, timeout=30)
    stream = json.loads(meta.stdout)['streams'][0]
    if stream['pix_fmt'] not in ('yuv444p', 'yuvj444p'):
        raise ValueError('Depth requires native 8-bit 4:4:4 planar channels; refusing conversion')
    h, w = int(stream['height']), int(stream['width'])
    nbytes = h*w*3
    if nbytes <= 0 or nbytes > 200_000_000:
        raise ValueError('Unsupported frame size')
    # Keep full-range/limited-range sample codes unchanged by using native format.
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:v:0',
           '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', stream['pix_fmt'], 'pipe:1']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    count = 0
    try:
        while True:
            blob = bytearray()
            while len(blob) < nbytes:
                part = proc.stdout.read(nbytes-len(blob))
                if not part: break
                blob.extend(part)
            if not blob: break
            if len(blob) != nbytes:
                raise ValueError('Truncated decoded depth frame')
            count += 1
            yield unpack_depth_yuv444(np.frombuffer(blob, np.uint8).reshape(3, h, w))
        if proc.wait(timeout=30) != 0:
            raise RuntimeError('ffmpeg depth decode failed')
        if expected_count is not None and count != expected_count:
            raise ValueError(f'Decoded frame count {count} != timestamps {expected_count}')
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(timeout=5)


def checked_times(values, name):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or a.size < 2 or not np.isfinite(a).all() or (np.diff(a) <= 0).any():
        raise ValueError(f'{name}: need finite, strictly increasing timestamps')
    return a


def associate_timestamps(color_us, depth_us, tolerance_us):
    """Mutual-nearest matching, ties to earlier index; never reuse a source frame.

    All RGB indices are retained in the result, including those without GT. A
    streaming evaluator must process every RGB frame, scoring only matched GT.
    Tolerance is supplied by the prospective data protocol, never optimized here.
    """
    color = checked_times(color_us, 'color')
    depth = checked_times(depth_us, 'depth')
    if not np.isfinite(tolerance_us) or tolerance_us < 0:
        raise ValueError('Non-negative finite tolerance required')

    def nearest(query, target):
        hi = np.clip(np.searchsorted(target, query), 0, len(target)-1)
        lo = np.maximum(hi-1, 0)
        return np.where(np.abs(query-target[lo]) <= np.abs(query-target[hi]), lo, hi)

    c_to_d, d_to_c = nearest(color, depth), nearest(depth, color)
    distance = np.abs(color-depth[c_to_d])
    keep = (d_to_c[c_to_d] == np.arange(len(color))) & (distance <= tolerance_us)
    rows = [{'color_index': i, 'color_us': float(t),
             'depth_index': int(c_to_d[i]) if keep[i] else None,
             'depth_us': float(depth[c_to_d[i]]) if keep[i] else None,
             'offset_us': float(depth[c_to_d[i]]-t) if keep[i] else None}
            for i, t in enumerate(color)]
    selected = [r['depth_index'] for r in rows if r['depth_index'] is not None]
    assert len(selected) == len(set(selected))
    return {'frames': rows, 'color_count': len(color), 'depth_count': len(depth),
            'matched': int(keep.sum()), 'coverage': float(keep.mean()),
            'tolerance_us': float(tolerance_us), 'rule': 'mutual nearest; ties earlier',
            'color_interval_us': {'median': float(np.median(np.diff(color))),
                                  'max': float(np.max(np.diff(color)))},
            'depth_interval_us': {'median': float(np.median(np.diff(depth))),
                                  'max': float(np.max(np.diff(depth)))},
            'all_color_frames_preserved': True, 'gt_admitted': False}
