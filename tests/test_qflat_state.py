import torch
from sokkanaem.qflat_state import flat_mask, flat_excess_loss, refresh_memory


def make_state():
    return {"features": [torch.rand(1, 5, 8)], "pooled": torch.rand(1, 8),
            "grid": (2, 2), "size": (28, 28), "h": torch.ones(4, 8, 8)}


def test_refresh_preserves_exact_features_and_propagates_h_gradients():
    old, fresh = make_state(), make_state()
    old["h"].requires_grad_(); fresh["h"].zero_()
    image = torch.rand(1, 3, 28, 28)
    out, n = refresh_memory(fresh, old, image, image, True)
    assert n == 4 and out["features"] is fresh["features"] and out["pooled"] is fresh["pooled"]
    torch.testing.assert_close(out["h"], old["h"]*.9)
    out["h"].sum().backward()
    torch.testing.assert_close(old["h"].grad, torch.full_like(old["h"], .9))


def test_large_change_and_hard_reset_discard_memory():
    old, fresh = make_state(), make_state(); fresh["h"].zero_()
    zero = torch.zeros(1, 3, 28, 28)
    out, n = refresh_memory(fresh, old, zero+1, zero, True)
    assert n == 0 and out["h"].abs().sum() == 0
    out, n = refresh_memory(fresh, old, zero, zero, True, hard_reset=True)
    assert out is fresh and n == 0


def test_flat_mask_and_excess_loss():
    gt = torch.ones(1, 1, 32, 32)*2
    mask, valid = flat_mask(gt, torch.ones_like(gt).bool())
    assert mask.sum() > 10
    assert flat_excess_loss(gt, gt, valid, mask) == 0
    pred = (gt+torch.rand_like(gt)*.1).requires_grad_()
    loss = flat_excess_loss(pred, gt, valid, mask)
    assert loss > 0
    loss.backward(); assert torch.isfinite(pred.grad).all()
    empty = torch.zeros_like(mask)
    assert flat_excess_loss(pred, gt, valid, empty) == 0
