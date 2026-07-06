"""Core-claim checks (IDEA.md §3.2): Δ-gating gives EXACT state copy,
and identical consecutive frames produce near-zero active ratio.

Run:  python -m pytest tests/ -q   (or: python tests/test_gating.py)
"""
import torch

from sokkanaem import SOKKANAEM, ChangeDetector, SelectiveSSM


def test_delta_gating_exact_state_copy():
    torch.manual_seed(0)
    ssm = SelectiveSSM(dim=32, d_state=8).eval()
    u = torch.randn(2, 32)
    h0 = torch.randn(2, 64, 8)  # d_inner = 2*32
    _, h1 = ssm.step(u, mask=torch.zeros(2), h=h0)
    assert torch.equal(h1, h0), "mask=0 must copy hidden state bit-exactly"
    _, h2 = ssm.step(u, mask=torch.ones(2), h=h0)
    assert not torch.equal(h2, h0), "mask=1 must update hidden state"


def test_static_scene_skips_and_depth_stable():
    torch.manual_seed(0)
    model = SOKKANAEM(keyframe_every=1000).eval()
    frame = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        d1, state, i1 = model.step(frame, None)          # first frame: full
        d2, state, i2 = model.step(frame.clone(), state)  # identical frame
    assert i1["active_ratio"] == 1.0
    assert i2["active_ratio"] == 0.0, "identical frame must be fully static"
    assert torch.allclose(d1, d2, atol=1e-5), "static scene must give stable depth"


def test_detector_hysteresis_and_keyframe():
    det = ChangeDetector(patch_size=16, tau_on=0.02, tau_off=0.01, keyframe_every=3)
    f = torch.zeros(1, 3, 64, 64)
    assert det(f).mean() == 1.0        # frame 0: keyframe
    assert det(f).mean() == 0.0        # frame 1: static
    f2 = f.clone()
    f2[:, :, :16, :16] = 1.0
    m = det(f2)                        # frame 2: one patch changed (+dilation)
    assert 0 < m.mean() < 1.0
    assert det(f2).mean() == 1.0       # frame 3: keyframe refresh


if __name__ == "__main__":
    test_delta_gating_exact_state_copy()
    test_static_scene_skips_and_depth_stable()
    test_detector_hysteresis_and_keyframe()
    print("all gating tests passed")
