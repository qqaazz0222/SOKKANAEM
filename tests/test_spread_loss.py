"""T5-33: the range-compression penalty must reward the thing it names.

REPORT 4.32 measured the defect this term exists for -- predictions with under
half the ground truth's dynamic range -- so the checks are about spread, not
about the value of the number: zero when the spread matches, positive when the
prediction is flattened, and symmetric so the term cannot be bought by
inflating the range with noise.
"""
import torch

from sokkanaem.losses import spread_loss


def _fields(compress=1.0, scale=1.0):
    """A depth field and a version whose log-spread is scaled by `compress`."""
    torch.manual_seed(0)
    lg = torch.randn(2, 1, 16, 16)
    lp = lg * compress
    return (lp.exp() * scale, lg.exp(), torch.ones_like(lg))


def test_zero_when_spread_matches():
    pred, gt, valid = _fields(compress=1.0)
    assert spread_loss(pred, gt, valid) < 1e-6


def test_scale_invariant():
    """A pure scale error is the other losses' business, not this one."""
    pred, gt, valid = _fields(compress=1.0, scale=7.0)
    assert spread_loss(pred, gt, valid) < 1e-6


def test_penalises_compression():
    pred, gt, valid = _fields(compress=0.47)   # the measured Bonn ratio
    assert spread_loss(pred, gt, valid) > 0.5


def test_symmetric_so_noise_cannot_buy_it():
    """Over-spreading must cost the same as under-spreading by the same factor,
    otherwise the cheapest way to lower this loss is to add noise."""
    lo = spread_loss(*_fields(compress=0.5))
    hi = spread_loss(*_fields(compress=2.0))
    assert torch.allclose(lo, hi, rtol=1e-4)


def test_ignores_samples_without_enough_valid_pixels():
    pred, gt, valid = _fields(compress=0.3)
    valid = torch.zeros_like(valid)
    valid[0, :, :1, :4] = 1          # 4 pixels: not a range estimate
    assert spread_loss(pred, gt, valid) == 0.0


def test_differentiable():
    pred, gt, valid = _fields(compress=0.5)
    pred.requires_grad_(True)
    spread_loss(pred, gt, valid).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))
