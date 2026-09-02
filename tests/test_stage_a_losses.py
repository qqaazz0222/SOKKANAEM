"""The two Stage A terms PLAN.md §6 asks for, and the directions they must have.

Both are written against measured defects, so the tests are about which
prediction each term prefers -- not about a value."""
import torch
import torch.nn.functional as F

from sokkanaem.losses import boundary_location_loss, rank_loss


def _scene(H=96, W=96):
    ramp = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(1, 1, H, W)
    gt = 8.0 - 4.0 * ramp.clone()
    gt[..., W // 2:] = 2.0 + 0.5 * ramp[..., W // 2:]
    gt[..., H // 4:H // 4 + 16, W // 4:W // 4 + 16] = 1.2
    return gt.contiguous(), torch.ones_like(gt)


def _blur(x, k=9):
    return F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="replicate"), k, stride=1)


def _rescaled(depth, a=3.0, b=0.4):
    return 1.0 / (a * (1.0 / depth) + b)


def test_boundary_loss_prefers_the_truth_over_blur_and_over_noise():
    gt, valid = _scene()
    torch.manual_seed(0)
    exact = boundary_location_loss(gt.clone(), gt, valid)
    blurred = boundary_location_loss(_blur(gt), gt, valid)
    noisy = boundary_location_loss(gt * (1 + 0.02 * torch.randn_like(gt)),
                                   gt, valid)
    assert exact < blurred and exact < noisy


def test_boundary_loss_does_not_reward_exceeding_the_gt_gradient():
    """The edge term is a hinge: sharpening past the truth must not pay, or the
    term buys the gradient ratio with ringing."""
    gt, valid = _scene()
    over = (gt + 2.0 * (gt - _blur(gt, 5))).clamp(min=0.2)
    # only the edge term can be gamed; isolate it by scoring inside the band
    e_exact = boundary_location_loss(gt.clone(), gt, valid)
    e_over = boundary_location_loss(over, gt, valid)
    assert e_over > e_exact


def test_boundary_loss_is_scale_and_shift_invariant():
    gt, valid = _scene()
    pred = _blur(gt)
    a = boundary_location_loss(pred, gt, valid)
    b = boundary_location_loss(_rescaled(pred), gt, valid)
    assert abs(float(a) - float(b)) < 1e-4


def test_rank_loss_is_zero_ish_when_ordering_is_right_and_large_when_inverted():
    gt, valid = _scene()
    torch.manual_seed(0)
    right = float(rank_loss(gt.clone(), gt, valid))
    torch.manual_seed(0)
    # invert the ordering while keeping the same value distribution
    inverted = gt.max() + gt.min() - gt
    wrong = float(rank_loss(inverted, gt, valid))
    # softplus never reaches 0 for a finite gap, so the level is not the
    # signal -- the ordering is
    assert wrong > 2 * right


def test_rank_loss_ignores_scale_and_shift():
    """Ordering is gauge-free, which is why this term can sit next to a metric
    loss without fighting it over scale."""
    gt, valid = _scene()
    pred = _blur(gt)
    torch.manual_seed(0)
    a = float(rank_loss(pred, gt, valid))
    torch.manual_seed(0)
    b = float(rank_loss(_rescaled(pred), gt, valid))
    assert abs(a - b) < 1e-3


def test_both_terms_backprop():
    gt, valid = _scene()
    pred = _blur(gt).requires_grad_(True)
    (boundary_location_loss(pred, gt, valid) + rank_loss(pred, gt, valid)).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0


def _pan_with_mover(T=4, H=160, W=160):
    """Camera pans right over a textured wall; a square moves left against it.
    Depth: wall far, square near."""
    xx = torch.linspace(0, 1, W * 2).view(1, W * 2).expand(H, W * 2)
    frames, gt = [], []
    for t in range(T):
        strip = xx[:, t * 4:t * 4 + W].clone()
        d = torch.full((1, H, W), 6.0)
        x0 = W // 2 - t * 6
        strip[40:80, x0:x0 + 32] = 0.9                 # the mover, textured
        d[:, 40:80, x0:x0 + 32] = 1.5
        frames.append(torch.stack([strip, strip * 0.5, 1 - strip]))
        gt.append(d)
    return torch.stack(frames)[None], torch.stack(gt)[None]


def test_dynamic_weight_costs_more_on_the_mover_than_on_the_wall():
    """Same error magnitude, different place: the moving object has to be the
    expensive one, or the term is not weighting what its name says."""
    from sokkanaem.losses import dynamic_weighted_loss

    frames, gt = _pan_with_mover()
    valid = torch.ones_like(gt)
    on_mover, on_wall = gt.clone(), gt.clone()
    on_mover[..., 40:80, 60:92] *= 1.3       # where the square is around t=1..2
    on_wall[..., 100:140, 10:42] *= 1.3      # static background patch
    a = float(dynamic_weighted_loss(frames, on_mover, gt, valid))
    b = float(dynamic_weighted_loss(frames, on_wall, gt, valid))
    assert a > b


def test_overshoot_loss_fires_on_ringing_and_not_on_blur():
    """Its whole job is to be the term blur cannot trip and ringing must."""
    from sokkanaem.losses import overshoot_loss

    gt, valid = _scene()
    ringing = (gt + 1.5 * (gt - _blur(gt, 5))).clamp(min=0.2)
    assert float(overshoot_loss(_blur(gt), gt, valid)) < 1e-4
    assert float(overshoot_loss(gt.clone(), gt, valid)) < 1e-4
    assert float(overshoot_loss(ringing, gt, valid)) > 1e-3


def test_overshoot_loss_backprops():
    from sokkanaem.losses import overshoot_loss

    gt, valid = _scene()
    pred = (gt + 1.5 * (gt - _blur(gt, 5))).clamp(min=0.2).requires_grad_(True)
    overshoot_loss(pred, gt, valid).backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) > 0
