import io
import zipfile
import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image
from scripts.fixed_camera_admission import complete_prefix_members, ur_depth_m, strict_pairs
from sokkanaem.qfixed_controls import PeriodicOutputHold


def test_ur_scale():
    assert ur_depth_m([0, 65535]).tolist() == pytest.approx([0, 3.640])
    assert ur_depth_m([65535], camera=0).item() == pytest.approx(6)
    assert ur_depth_m([65535], camera=0, adl=True).item() == pytest.approx(7)
    with pytest.raises(ValueError): ur_depth_m([1], camera=1, adl=True)


def test_pair_ids_and_missing():
    assert strict_pairs(['r000002.png','r000001.png'], ['d000001.png','d000002.png'])[0][0] == 1
    with pytest.raises(ValueError): strict_pairs(['r1.png'], ['d2.png'])
    with pytest.raises(ValueError): strict_pairs(['r1.png','r3.png'], ['d1.png','d3.png'])
    with pytest.raises(ValueError): strict_pairs(['a1.png','b1.png'], ['d1.png'])


def test_crc_checked_prefix(tmp_path):
    b = io.BytesIO()
    Image.fromarray(np.full((3,4), 60000, np.uint16)).save(b, format='PNG')
    path = tmp_path/'full.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('depth1.png', b.getvalue())
    blob = path.read_bytes()
    prefix = tmp_path/'prefix.zip'
    prefix.write_bytes(blob[:blob.index(b'PK\x01\x02')])
    rows = complete_prefix_members(prefix)
    assert len(rows) == 1 and (rows[0][1] == 60000).all()
    prefix.write_bytes(blob[:40])
    assert complete_prefix_members(prefix) == []


class Exact(nn.Module):
    def step(self, frame, state=None):
        return frame[:, :1].clone(), None, {}


def test_periodic_hold_refresh_and_reset():
    model = PeriodicOutputHold(Exact(), period=2)
    state = None
    for t in range(6):
        pred, state, info = model.step(torch.full((1,3,4,4), float(t)), state)
        assert torch.all(pred == t-t%2)
        assert info['full_refresh'] == (t%2 == 0)
        assert info['flow_calls'] == 0
    _, _, info = model.step(torch.zeros(1,3,5,5), state)
    assert info['full_refresh']
    with pytest.raises(ValueError): PeriodicOutputHold(Exact(), period=0)
