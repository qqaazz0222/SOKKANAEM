import pytest
import torch
from sokkanaem.bounded_refiner import BoundedDepthRefiner, refinement_loss
from sokkanaem.edge_local_refiner import EdgeLocalRefiner, edge_local_loss


def test_identity_and_bound():
    m = BoundedDepthRefiner()
    rgb, d, f = torch.rand(2, 3, 16, 16), torch.rand(2, 1, 16, 16) + .3, torch.rand(2, 16, 8, 8)
    p, delta = m(rgb, d, f)
    assert torch.equal(p, d) and torch.count_nonzero(delta) == 0
    with torch.no_grad():
        m.head.weight.normal_(0, 50)
    p, delta = m(rgb, d, f)
    assert delta.abs().max() <= .15
    assert torch.isfinite(p).all() and (p > 0).all()


@pytest.mark.parametrize("empty", [False, True])
def test_loss_backward_and_invalid_gt(empty):
    m = BoundedDepthRefiner()
    rgb, d, f = torch.rand(2, 3, 16, 16), torch.ones(2, 1, 16, 16), torch.rand(2, 16, 8, 8)
    p, delta = m(rgb, d, f)
    gt = d * 1.1
    valid = torch.ones_like(d) if not empty else torch.zeros_like(d)
    gt[..., 0, :] = float("nan")
    gt[..., 1, :] = float("inf")
    loss, _ = refinement_loss(p, d, gt, valid, delta)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(m.head.weight.grad).all()


def test_local_identity_bound_and_flat_gate():
    m = EdgeLocalRefiner()
    rgb, d, f = torch.rand(2, 3, 16, 16), torch.ones(2, 1, 16, 16), torch.rand(2, 16, 8, 8)
    p, delta = m(rgb, d, f)
    assert torch.equal(p, d)
    with torch.no_grad():
        m.head.weight.normal_(0, 30)
    p, delta = m(rgb, d, f)
    assert torch.equal(p, d), "RGB texture cannot change a constant coarse depth"
    d[..., 8:] *= 2
    p, delta = m(rgb, d, f)
    assert delta.abs().max() <= .15
    gt = d.clone(); gt[..., 0, :] = float("nan")
    loss, _ = edge_local_loss(p, d, gt, torch.isfinite(gt), delta)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(m.head.weight.grad).all()
