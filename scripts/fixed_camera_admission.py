"""Reusable format checks; never fit depth scale or registration to predictions."""
import io
import re
import struct
import zlib
from pathlib import Path
import numpy as np
from PIL import Image


def complete_prefix_members(path):
    """Read CRC-verified local entries from an incomplete ZIP prefix.

    Returned images are forensic samples, not a recovered full dataset.
    Data-descriptor/ZIP64 cases are rejected rather than guessed.
    """
    blob = Path(path).read_bytes()
    pos = 0
    rows = []
    while pos + 30 <= len(blob):
        fields = struct.unpack_from('<IHHHHHIIIHH', blob, pos)
        magic, version, flags, method, tm, dt, crc, csize, size, nlen, xlen = fields
        if magic != 0x04034b50: break
        if flags & 9 or csize == 0xffffffff: break
        start = pos + 30 + nlen + xlen
        end = start + csize
        if end > len(blob): break
        name = blob[pos+30:pos+30+nlen].decode('utf-8')
        payload = blob[start:end]
        if method == 8: payload = zlib.decompress(payload, -15)
        elif method != 0: raise ValueError('Unsupported ZIP compression')
        assert len(payload) == size and zlib.crc32(payload) == crc
        if name.lower().endswith('.png'):
            rows.append((name, np.array(Image.open(io.BytesIO(payload)))))
        pos = end
    return rows


def ur_depth_m(values, camera=1, adl=False):
    if camera not in (0, 1) or (adl and camera != 0):
        raise ValueError('Undocumented camera/event combination')
    scale = 7000 if adl else (6000 if camera == 0 else 3640)
    return np.asarray(values, dtype=np.float64) * (scale / 65535 / 1000)


def indexed_frames(paths):
    result = {}
    for p in paths:
        match = re.search(r'(\d+)$', Path(p).stem)
        if not match: raise ValueError(f'Missing frame ID: {p}')
        i = int(match.group(1))
        if i in result: raise ValueError(f'Duplicate frame ID: {i}')
        result[i] = str(p)
    return result


def strict_pairs(rgb, depth):
    r, d = indexed_frames(rgb), indexed_frames(depth)
    if not r or r.keys() != d.keys():
        raise ValueError('RGB/depth frame IDs differ')
    ids = sorted(r)
    if ids != list(range(ids[0], ids[-1]+1)):
        raise ValueError('Non-contiguous frames')
    return [(i, r[i], d[i]) for i in ids]
