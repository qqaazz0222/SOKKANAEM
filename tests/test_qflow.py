import cv2
import numpy as np
import pytest
import torch
from sokkanaem.qflow import correspondence, warp_depth, RGBFlowStream


class FixedFlow:
    def __init__(self, backward, forward):
        self.flows = iter((backward, forward))
    def calc(self, *args):
        return next(self.flows).copy()


class Exact:
    def step(self, frame):
        return frame[:, :1]+1, None, {"full_refresh": True, "backbone_calls": 1}


def test_warp_direction_scaling_and_bounds():
    depth = torch.arange(64).float().reshape(1, 1, 8, 8)+1
    flow = np.zeros((4, 4, 2), np.float32); flow[..., 0] = 1
    out, mask = warp_depth(depth, flow, np.ones((4, 4), bool))
    assert torch.equal(out[..., :-2], depth[..., 2:])
    assert mask[..., :-2].all() and not mask[..., -2:].any()


def test_subpixel_modes_are_distinct():
    depth = torch.arange(64).float().reshape(1, 1, 8, 8)
    flow = np.zeros((8, 8, 2), np.float32); flow[..., 0] = .25
    mask = np.ones((8, 8), bool)
    nearest, _ = warp_depth(depth, flow, mask, "nearest")
    linear, _ = warp_depth(depth, flow, mask, "bilinear")
    assert torch.equal(nearest, depth)
    torch.testing.assert_close(linear[..., :-1], depth[..., :-1]+.25)


def test_confidence_rejects_inconsistent_photometric_and_boundary_pixels():
    img = np.zeros((12, 12), np.uint8); zero = np.zeros((12, 12, 2), np.float32)
    _, reliable = correspondence(img, img, FixedFlow(zero, zero))
    assert reliable[1:-1, 1:-1].all() and not reliable[0].any()
    _, reliable = correspondence(img, img+255, FixedFlow(zero, zero))
    assert not reliable.any()
    _, reliable = correspondence(img, img, FixedFlow(zero, zero+3))
    assert not reliable.any()


def test_k2_exact_refresh_and_no_caller_state_mutation():
    cv2.setNumThreads(1)
    m = RGBFlowStream(Exact(), gated=True, size=64)
    frame = torch.rand(1, 3, 64, 64)
    d0, state, i0 = m.step(frame); original = state["depth"].clone()
    d1, next_state, i1 = m.step(frame, state)
    assert i0["full_refresh"] and not i1["full_refresh"]
    assert torch.equal(d0, d1) and torch.equal(state["depth"], original)
    assert next_state["depth"].data_ptr() != state["depth"].data_ptr()
    d2, _, i2 = m.step(frame*.5, next_state)
    assert i2["full_refresh"] and torch.equal(d2, frame[:, :1]*.5+1)


def test_low_confidence_forces_full_refresh_and_shape_reset():
    m = RGBFlowStream(Exact(), gated=True, max_unreliable=0, size=64)
    frame = torch.rand(1, 3, 64, 64)
    _, state, _ = m.step(frame)
    depth, _, info = m.step(frame, state)
    assert info["reason"] == "confidence" and torch.equal(depth, frame[:, :1]+1)
    _, _, info = m.step(frame[:, :, :60, :60], state)
    assert info["reason"] == "initial"
    with pytest.raises(ValueError): m.step(frame.expand(2, -1, -1, -1))


def test_actual_dis_backward_translation_sign():
    rng = np.random.default_rng(91)
    previous = cv2.GaussianBlur(rng.integers(0, 256, (128, 128), dtype=np.uint8), (3, 3), 0)
    current = cv2.warpAffine(previous, np.float32([[1, 0, 3], [0, 1, 0]]), (128, 128), borderMode=cv2.BORDER_REFLECT)
    engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    flow, reliable = correspondence(previous, current, engine)
    assert abs(np.median(flow[24:-24, 24:-24, 0])+3) < .6
    assert reliable[24:-24, 24:-24].mean() > .8
