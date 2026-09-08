import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.paper_edge_input_bench import crop, sha, verify_bundle
from sokkanaem.performance_study import rgb_view


def test_edge_crop_matches_frozen_roi():
    image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (480, 640, 3), dtype=np.uint8))
    assert np.array_equal(np.asarray(crop(image)), np.asarray(rgb_view(image)))


def test_edge_hash_tampering_and_escape_rejected(tmp_path):
    image = tmp_path/'frame.png'
    image.write_bytes(b'original')
    doc = dict(files=[dict(path='frame.png', sha256=sha(image))])
    verify_bundle(tmp_path, doc)
    image.write_bytes(b'changed')
    with pytest.raises(ValueError):
        verify_bundle(tmp_path, doc)
    doc['files'][0]['path'] = '../frame.png'
    with pytest.raises(ValueError):
        verify_bundle(tmp_path, doc)
