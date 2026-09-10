import torch
from sokkanaem.qquality import align_state, depth_retention, motion_grid, decode_with_grad
from sokkanaem.qdelta_ssm import QDeltaSSM


def test_depth_retention_exact_and_gradient():
    target = torch.rand(1, 1, 16, 16)+1
    a, b = depth_retention(target, target)
    assert a == 0 and b == 0
    pred = (target*.9).requires_grad_()
    a, b = depth_retention(pred, target)
    (a+b).backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_identity_motion_and_state_transport():
    torch.manual_seed(41)
    rgb = torch.rand(1, 3, 16, 16)
    grid, cost = motion_grid(rgb, rgb, (4, 4), match_size=16)
    m = QDeltaSSM(channels=8, stages=2, width=8)
    state = m.refresh({"features": [torch.rand(1, 17, 8) for _ in range(2)],
                       "pooled": torch.rand(1, 8), "grid": (4, 4), "size": (56, 56)})
    state["h"].normal_()
    out = align_state(state, grid)
    for a, b in zip(out["features"], state["features"]):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(out["h"], state["h"])
    assert torch.equal(out["pooled"], state["pooled"]) and cost.max() == 0


def test_motion_direction_for_translation():
    torch.manual_seed(7)
    previous = torch.rand(1, 3, 20, 20)
    current = torch.roll(previous, 1, dims=-1)
    grid, _ = motion_grid(previous, current, (20, 20), match_size=20)
    x = (torch.arange(20)+.5)*2/20-1
    # A current pixel at x must sample the previous pixel at x-1.
    torch.testing.assert_close(grid[0, 4:-4, 4:-4, 0], x[3:-5].expand(12, -1))


def test_decoder_gradients_do_not_require_trainable_q0():
    class Q:
        def _decode(self, features, grid): return features[0]*2
        def _calibrate(self, disp, pooled, size): return disp+pooled
    class Exact: q = Q()
    x = torch.ones(1, 1, 8, 8, requires_grad=True)
    y = decode_with_grad(Exact(), {"features": [x], "grid": (2, 2), "pooled": 1, "size": (8, 8)})
    y.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2))


def test_rgb_refiner_identity_bound_and_gradient():
    from sokkanaem.qrgb_refiner import QRGBRefiner
    m = QRGBRefiner(); d = torch.rand(1, 1, 16, 16)+1; rgb = torch.rand(1, 3, 16, 16)
    torch.testing.assert_close(m(rgb, d), d, rtol=0, atol=0)
    with torch.no_grad(): m.net[-1].bias.fill_(2.)
    out = m(rgb, d)
    assert ((out/d).log().abs() <= .060001).all()
    out.mean().backward()
    assert m.net[-1].weight.grad.abs().sum() > 0


def test_refiner_does_not_change_full_refresh():
    from sokkanaem.qrgb_refiner import QRGBRefiner, RefinedStream
    class Stream(torch.nn.Module):
        def step(self, frame, state=None):
            return frame[:, :1]+1, True, {"full_refresh": state is None}
    m = QRGBRefiner()
    with torch.no_grad(): m.net[-1].bias.fill_(2.)
    stream = RefinedStream(Stream(), m); rgb = torch.rand(1, 3, 16, 16)
    first, state, _ = stream.step(rgb)
    second, _, _ = stream.step(rgb, state)
    assert torch.equal(first, rgb[:, :1]+1) and (second > first).all()


def test_risk_features_identical_frames():
    from sokkanaem.qrisk import risk_features, RefreshRisk
    frame = torch.rand(1, 3, 37, 37)
    x = risk_features(frame, frame, 1)
    assert x.shape == (1, 14) and torch.equal(x[:, :6], torch.zeros(1, 6))
    assert x[0, -1] == 0
    assert RefreshRisk()(x).item() > 0


def test_risk_refresh_and_bounded_age():
    from sokkanaem.qrisk import RiskStream
    class Exact(torch.nn.Module):
        def encode(self, frame): return {"depth": frame[:, :1]+1}
        def decode(self, state): return state["depth"]
    class Adapter(torch.nn.Module):
        def refresh(self, state): return state
        def update(self, frame, mask, state): return state, {"updated_patches": int(mask.sum())}
    class Risk(torch.nn.Module):
        def __init__(self, value): super().__init__(); self.value = value
        def forward(self, x): return x.new_full((1, 1), self.value)
    frame = torch.rand(1, 3, 32, 32)
    for value, expected in ((1., [True]*6), (0., [True, False, False, False, True, False])):
        model = RiskStream(Exact(), Adapter(), Risk(value), threshold=.02)
        state = None; got = []
        for _ in range(6):
            _, state, info = model.step(frame, state); got.append(info["full_refresh"])
        assert got == expected
