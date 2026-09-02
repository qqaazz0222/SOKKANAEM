"""PLAN_ACC A7: the near-range term must move where the loss looks, not how
loud it is.

REPORT §4.43 (A6) put our close-range error at 2.38x a 343M reference while
everything outside the band sits at ~1.13x. The term exists to redirect
gradient into that band, so the properties worth pinning are: an error inside
the band costs more than the same error outside it, the renormalization keeps
the term's overall magnitude comparable across scenes, and invalid pixels
never enter the fit.
"""
import torch

from sokkanaem.losses import near_weighted_loss

NEAR, FAR = 1.0, 5.0     # metres, either side of the 2.0 m default band


def _scene(err_near=0.0, err_far=0.0):
    """Half the pixels close, half far; a multiplicative error on each half."""
    gt = torch.cat([torch.full((1, 1, 4, 8), NEAR),
                    torch.full((1, 1, 4, 8), FAR)], dim=-2)
    pred = gt.clone()
    pred[..., :4, :] *= 1.0 + err_near
    pred[..., 4:, :] *= 1.0 + err_far
    return pred, gt, torch.ones_like(gt)


def test_zero_on_a_perfect_prediction():
    assert near_weighted_loss(*_scene()) < 1e-6


def test_near_error_costs_more_than_the_same_error_far():
    """Same relative error, different band -- that ratio IS the term."""
    near = near_weighted_loss(*_scene(err_near=0.2))
    far = near_weighted_loss(*_scene(err_far=0.2))
    assert near > far, (near, far)
    assert abs(near / far - 4.0) < 0.05, (near, far)   # the default gain


def test_gain_one_is_plain_log_l1():
    """With no gain the weight is uniform, so the band cannot matter."""
    a = near_weighted_loss(*_scene(err_near=0.2), gain=1.0)
    b = near_weighted_loss(*_scene(err_far=0.2), gain=1.0)
    assert abs(a - b) < 1e-6, (a, b)


def test_scale_free_in_magnitude():
    """Weights renormalize to mean 1, so a scene that is ALL near-range must
    not be penalised harder just for being close -- otherwise the term would
    silently reweight datasets instead of pixels."""
    gt = torch.full((1, 1, 8, 8), NEAR)
    v = torch.ones_like(gt)
    a = near_weighted_loss(gt * 1.2, gt, v)
    gt_far = torch.full((1, 1, 8, 8), FAR)
    b = near_weighted_loss(gt_far * 1.2, gt_far, v)
    assert abs(a - b) < 1e-6, (a, b)


def test_invalid_pixels_do_not_enter():
    """A Kinect dropout reads 0 m -- inside any near band, and infinitely
    wrong. It must not become the loudest pixel in the batch."""
    pred, gt, valid = _scene(err_far=0.2)
    gt = gt.clone()
    gt[..., :4, :] = 0.0
    valid = valid.clone()
    valid[..., :4, :] = 0.0
    out = near_weighted_loss(pred, gt, valid)
    assert torch.isfinite(out) and out > 0, out
    # only the far half is left, so the band never fires and the weight is flat
    assert abs(out - near_weighted_loss(pred, gt, valid, gain=1.0)) < 1e-6
