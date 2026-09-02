"""The four properties PLAN.md section 5 requires of a sharpness evaluator.

A gradient ratio on its own is buyable with noise, so these tests pin the
directions: blur must lose F1 *and* ratio, noise may win the ratio but must
lose on flat TV and precision, and none of it may move under a scale+shift of
the prediction."""
import torch
import torch.nn.functional as F

from sokkanaem.sharpness import sharpness_scores


def _scene(H=96, W=96):
    """Two slabs and a small square over a ramp: real depth boundaries, and
    interiors that are smooth rather than exactly constant -- a piecewise
    constant scene has a zero 90th-percentile gradient, which no threshold
    rule can read."""
    ramp = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(1, 1, H, W)
    gt = 8.0 - 4.0 * ramp.clone()                     # 8 m -> 4 m background
    gt[..., W // 2:] = 2.0 + 0.5 * ramp[..., W // 2:]  # near slab, still sloped
    gt[..., H // 4:H // 4 + 16, W // 4:W // 4 + 16] = 1.2
    return gt.contiguous(), torch.ones_like(gt)


def _blur(x, k=9):
    return F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="replicate"), k, stride=1)


def _rescaled(depth, a=3.0, b=0.4):
    """Prediction with its disparity scaled and shifted -- exactly what an
    alignment-free relative-shape metric must ignore."""
    return 1.0 / (a * (1.0 / depth) + b)


def test_perfect_prediction_is_optimal():
    gt, valid = _scene()
    s = sharpness_scores(gt.clone(), gt, valid)
    assert s["boundary_f1"] > 0.99
    assert abs(s["grad_ratio"] - 1.0) < 1e-3
    assert abs(s["flat_tv_ratio"] - 1.0) < 1e-3
    assert s["overshoot"] < 1e-6
    assert s["absrel_edge"] < 1e-6


def test_blur_loses_ratio_and_f1_together():
    gt, valid = _scene()
    s = sharpness_scores(_blur(gt), gt, valid)
    assert s["grad_ratio"] < 0.7
    assert s["boundary_f1"] < 0.6
    # a 9px box blur still clears the loosest (q=0.90) threshold, so recall
    # falls rather than vanishes -- what matters is the direction
    assert s["boundary_recall"] < 0.8      # cannot reach the GT threshold
    assert s["overshoot"] < 0.05           # blur does not ring


def test_noise_may_win_the_ratio_but_fails_flat_tv_and_precision():
    gt, valid = _scene()
    torch.manual_seed(0)
    # high-frequency, small in depth terms: AbsRel barely moves
    noisy = gt * (1 + 0.02 * torch.randn_like(gt))
    s = sharpness_scores(noisy, gt, valid)
    clean = sharpness_scores(gt.clone(), gt, valid)
    assert s["grad_ratio"] > clean["grad_ratio"]        # the loophole
    assert s["flat_tv_ratio"] > 3 * clean["flat_tv_ratio"]   # and it is caught
    assert s["boundary_precision"] < 0.5


def test_scale_and_shift_do_not_move_relative_shape_metrics():
    gt, valid = _scene()
    pred = _blur(gt)
    a = sharpness_scores(pred, gt, valid)
    b = sharpness_scores(_rescaled(pred), gt, valid)
    for k in ("grad_ratio", "boundary_f1", "boundary_precision",
              "boundary_recall", "flat_tv", "flat_tv_ratio", "overshoot"):
        assert abs(a[k] - b[k]) < 1e-3, k


def test_holes_do_not_become_boundaries():
    """Invalid pixels used to dominate the top gradient decile; a prediction
    that is exactly right must still score 1.0 with a hole in the mask."""
    gt, valid = _scene()
    valid[..., 40:56, 10:26] = 0
    s = sharpness_scores(gt.clone(), gt, valid)
    assert s["boundary_f1"] > 0.99
    assert abs(s["grad_ratio"] - 1.0) < 1e-3


def test_oversharpening_is_caught_by_overshoot():
    """The other way to buy a gradient ratio: ring the edges. Overshoot is the
    term that has to notice, since flat regions stay quiet."""
    gt, valid = _scene()
    blurred = _blur(gt, 5)
    ringing = (gt + 1.5 * (gt - blurred)).clamp(min=0.2)
    s = sharpness_scores(ringing, gt, valid)
    assert s["grad_ratio"] > 1.0
    assert s["overshoot"] > 0.1
