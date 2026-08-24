"""Gradient distillation must transfer shape without pinning the depth range.

That distinction is the whole reason this term exists: `affine_invariant_loss`
matches the teacher's values and was measured to hurt both domains, the
diagnosis being that it drags far-range structure. A gradient term is blind to
any monotone affine gauge, so it cannot do that.
"""
import torch

from sokkanaem.distill import teacher_grad_loss


def _scene(size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = 2.0 + torch.rand(1, 1, size, size, generator=g)
    d[..., 16:48, 16:48] = 6.0                      # an object, i.e. a boundary
    return d


def test_blind_to_scale_and_shift():
    """The same shape under a different affine gauge must score the same."""
    d = _scene()
    disp = 1.0 / d
    base = float(teacher_grad_loss(d, disp))
    for s, b in ((3.0, 0.0), (1.0, 0.5), (0.2, -0.1)):
        assert abs(float(teacher_grad_loss(d, s * disp + b)) - base) < 1e-4


def test_penalises_a_flattened_boundary():
    """A prediction that smooths the object edge away must score worse."""
    d = _scene()
    disp = 1.0 / d
    import torch.nn.functional as F
    blurred = 1.0 / F.avg_pool2d(disp, 9, stride=1, padding=4)
    assert float(teacher_grad_loss(blurred, disp)) > float(teacher_grad_loss(d, disp))


def test_leaves_the_range_free():
    """Compressing the predicted range but keeping the shape must cost less
    than destroying the shape -- this is what `affine_invariant_loss` could not
    do, and why it fought the ground truth over far-range structure."""
    d = _scene()
    disp = 1.0 / d
    lg = d.clamp(min=1e-3).log()
    compressed = (lg.mean() + (lg - lg.mean()) * 0.5).exp()
    import torch.nn.functional as F
    blurred = 1.0 / F.avg_pool2d(disp, 9, stride=1, padding=4)
    assert float(teacher_grad_loss(compressed, disp)) < \
        float(teacher_grad_loss(blurred, disp))
