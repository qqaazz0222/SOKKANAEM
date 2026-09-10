import pytest
import torch
from sokkanaem.qaligned_adapter import AlignedAdapter, zero_state_formula


def encoded():
    return {"features": [torch.randn(1, 5, 384) for _ in range(4)], "pooled": torch.randn(1, 384),
            "grid": (2, 2), "size": (28, 28)}


def test_common_initialization_and_near_equal_parameter_budget():
    torch.manual_seed(738); a = AlignedAdapter("ssm")
    torch.manual_seed(738); b = AlignedAdapter("mlp")
    for name in ("rgb", "context", "fuse", "readout"):
        for x, y in zip(getattr(a, name).parameters(), getattr(b, name).parameters()):
            assert torch.equal(x, y)
    assert sum(p.numel() for p in a.parameters()) == 89040
    assert sum(p.numel() for p in b.parameters()) == 89037


@pytest.mark.parametrize("mode", ("ssm", "mlp"))
def test_initial_output_identity_and_selective_copy(mode):
    model = AlignedAdapter(mode); state = model.refresh(encoded())
    frame = torch.rand(1, 3, 28, 28); mask = torch.tensor([[1., 0., 1., 0.]])
    out, info = model.update(frame, mask, state)
    for a, b in zip(state["features"], out["features"]): assert torch.equal(a, b)
    assert out["pooled"] is state["pooled"] and info["updated_patches"] == 2
    with torch.no_grad(): model.readout.weight.fill_(.01)
    out, _ = model.update(frame, mask, state)
    for a, b in zip(state["features"], out["features"]):
        assert torch.equal(a[:, [0, 2, 4]], b[:, [0, 2, 4]])
    assert torch.equal(state["h"][[1, 3]], out["h"][[1, 3]])
    sum(f.sum() for f in out["features"]).backward()
    assert torch.isfinite(model.readout.weight.grad).all()


def test_zero_state_ssm_is_nonrecurrent_formula_but_nonzero_history_matters():
    core = AlignedAdapter("ssm").ssm
    u = torch.randn(6, 32, requires_grad=True)
    actual, h = core.step(u)
    direct, hd = zero_state_formula(core, u)
    torch.testing.assert_close(actual, direct, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(h, hd, rtol=1e-5, atol=1e-6)
    actual.sum().backward()
    assert core.A_log.grad is None or not core.A_log.grad.any()
    carried, _ = core.step(u, h=torch.ones_like(h))
    assert not torch.allclose(carried, direct)
