"""§3.5 hybrid path: Low-Res GMC + feature-level temporal gating.

Claims verified:
1. GMC recovers a known global translation (warped prev ≈ current).
2. Under pure camera pan, the feature gate keeps most patches static,
   while the pixel detector (no GMC) would mark nearly everything active.
3. A locally moving object still comes out active after compensation.
"""
import pytest
import torch
import torch.nn.functional as F

cv2 = pytest.importorskip("cv2")

from sokkanaem import SOKKANAEM, GlobalMotionCompensator


def textured_frame(size=256, seed=0):
    """Smooth random texture — corners to track without the aliasing a
    hard-edged synthetic pattern causes at GMC's low resolution."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, size // 8, size // 8, generator=g)
    return F.interpolate(base, (size, size), mode="bilinear",
                         align_corners=False)


def shift(img, dx, dy):
    return torch.roll(img, shifts=(dy, dx), dims=(2, 3))


def test_gmc_recovers_translation():
    prev = textured_frame()
    cur = shift(prev, 12, 7)  # camera pan
    warped = GlobalMotionCompensator()(prev, cur)
    # interior only: roll wraps, warp zero-pads — borders differ by design
    sl = slice(32, -32)
    err = (warped - cur)[..., sl, sl].abs().mean().item()
    raw = (prev - cur)[..., sl, sl].abs().mean().item()
    assert err < 0.05 * raw, f"warp error {err:.4f} vs raw {raw:.4f}"


def test_feature_gate_skips_under_pan():
    torch.manual_seed(0)
    model = SOKKANAEM(gmc=True, tau_on=0.1, tau_off=0.05,
                      keyframe_every=1000).eval()
    prev = textured_frame()
    cur = shift(prev, 12, 7)
    # paste a moving object into cur (32x32 block, feature-space novel)
    cur[..., 96:128, 96:128] = torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        _, state, info = model.step(prev)
        assert info["active_ratio"] == 1.0  # first frame = keyframe
        _, _, info = model.step(cur, state)

    mask = info["mask"].view(16, 16)
    # object patches (rows/cols 6-7 + dilation ring) must be active
    assert mask[6:8, 6:8].min() == 1.0, "moving object missed"
    # interior static patches gated off despite full-frame pan. Borders are
    # excluded: roll wraps content there (and real pans do reveal new content
    # at frame edges — border-active is correct behavior, not a miss).
    interior = mask[3:13, 3:13].clone()
    interior[2:6, 2:6] = 0  # cut out object + dilation ring
    assert interior.mean() < 0.15, \
        f"gate failed under ego-motion: interior active {interior.mean():.2f}"


def test_pixel_detector_saturates_under_pan():
    """Baseline: without GMC the same pan activates almost everything."""
    torch.manual_seed(0)
    model = SOKKANAEM(gmc=False, keyframe_every=1000).eval()
    prev = textured_frame()
    with torch.no_grad():
        _, state, _ = model.step(prev)
        _, _, info = model.step(shift(prev, 12, 7), state)
    assert info["active_ratio"] > 0.9
