"""The properties the ACC comparison rests on (PLAN_ACC A2, §6).

If alignment is not exactly one function for every model, the table compares
protocols rather than models -- which already happened once here, on t-delta
(REPORT §4.10). These tests pin the three things that must hold for a row to
mean anything: each gauge undoes exactly the freedom it fits, a fit that sends
valid pixels through zero disparity is reported rather than clamped away, and a
metric model and a relative model reach the scorer through the same code.
"""
import pytest
import torch

from sokkanaem.alignment import EPS, align


def _scene(T=3, H=8, W=8, seed=0):
    torch.manual_seed(seed)
    gt = torch.rand(T, 1, H, W) * 4 + 0.5      # 0.5 .. 4.5 m
    return gt, torch.ones_like(gt)


def test_median_gauge_undoes_a_scale():
    gt, v = _scene()
    out, info = align(gt * 3.7, gt, v, "median", "depth")
    assert (out - gt).abs().max() < 1e-4
    assert not info["failed"]


def test_scale_gauge_undoes_a_disparity_scale():
    gt, v = _scene()
    out, info = align(1.0 / gt * 0.21, gt, v, "scale", "disparity")
    assert (out - gt).abs().max() < 1e-3, info


def test_scaleshift_gauge_undoes_a_disparity_affine():
    gt, v = _scene()
    out, info = align(1.0 / gt * 0.21 + 0.4, gt, v, "scaleshift", "disparity")
    assert (out - gt).abs().max() < 1e-3, info
    assert not info["failed"]


def test_one_dof_gauges_keep_positive_predictions_positive():
    """Positive inputs and positive GT admit a positive scale-only fit."""
    gt, v = _scene()
    for mode, space, pred in (("median", "depth", gt * 2),
                              ("scale", "disparity", 1.0 / gt + 0.9)):
        _, info = align(pred, gt, v, mode, space)
        assert info["neg_frac"] == 0.0 and not info["failed"], (mode, info)


def test_zero_crossing_is_reported_not_hidden():
    """GT disparity 1,1,10 against a prediction of 0,1,2 fits s=4.5, b=-0.5:
    the first pixel lands at disparity -0.5, which is not a depth. Clamping it
    reports 1000 m and lets the mean absorb it -- that is DA V2 Small's 0.2256
    against its 0.0967 clip median (PLAN_ACC §1.2)."""
    gt = torch.tensor([1.0, 1.0, 0.1]).view(3, 1, 1, 1)
    pred = torch.tensor([0.0, 1.0, 2.0]).view(3, 1, 1, 1)
    out, info = align(pred, gt, torch.ones_like(gt), "scaleshift", "disparity")
    assert info["failed"]
    assert abs(info["neg_frac"] - 1 / 3) < 1e-6, info
    assert abs(info["s"] - 4.5) < 1e-6 and abs(info["b"] + 0.5) < 1e-6, info
    assert abs(out[0].item() - 1.0 / EPS) < 1e-3, out


def test_metric_and_relative_models_share_one_path():
    """The same prediction expressed as depth or as its disparity twin must
    align identically: `space` is a statement about the input's units, not a
    second implementation. This is what lets one table hold ours and DPT."""
    gt, v = _scene(seed=1)
    depth = gt * 1.3 + 0.2
    for mode in ("scale", "scaleshift"):
        a, ia = align(depth, gt, v, mode, "depth")
        b, ib = align(1.0 / depth, gt, v, mode, "disparity")
        assert (a - b).abs().max() < 1e-4, (mode, ia, ib)
        assert abs(ia["s"] - ib["s"]) < 1e-6 and abs(ia["b"] - ib["b"]) < 1e-6


def test_clip_with_no_valid_gt_has_no_gauge():
    gt, _ = _scene()
    assert align(gt, gt, torch.zeros_like(gt), "median") is None
    assert align(gt, gt, torch.zeros_like(gt), "scaleshift") is None


def test_partial_validity_fits_on_valid_pixels_only():
    """An invalid pixel must not steer the fit -- a Kinect dropout reads 0 m,
    which is infinite disparity and would dominate any least squares."""
    gt, v = _scene(seed=2)
    gt[:, :, :, :4] = 0.0
    v[:, :, :, :4] = 0.0
    out, info = align(1.0 / gt.clamp(min=EPS) * 0.5 + 0.1, gt, v,
                      "scaleshift", "disparity")
    m = v.bool()
    assert (out[m] - gt[m]).abs().max() < 1e-3, info


def test_metric_panel_does_not_fit_ground_truth():
    gt, valid = _scene()
    pred = gt * 2
    for per_frame in (False, True):
        out, info = align(pred, gt, valid, "none", "depth", per_frame)
        assert torch.equal(out, pred)
        assert info["s"] == 1 and info["b"] == 0
        assert torch.allclose(((out - gt).abs() / gt).mean(), torch.tensor(1.0))
    scaled, _ = align(pred, gt, valid, "median", "depth")
    assert torch.allclose(scaled, gt)


def test_metric_panel_rejects_relative_disparity():
    gt, valid = _scene()
    with pytest.raises(ValueError, match="metric depth"):
        align(1 / gt, gt, valid, "none", "disparity")


def test_nonpositive_metric_prediction_is_reported_without_clipping():
    gt, valid = _scene()
    pred = -torch.ones_like(gt)
    out, info = align(pred, gt, valid, "none", "depth")
    assert torch.equal(out, pred)
    assert info["failed"] and info["neg_frac"] == 1
    _, info = align(pred, gt, valid, "median", "depth")
    assert info["failed"] and info["neg_frac"] == 1


@pytest.mark.parametrize("mode", ["none", "median", "scale", "scaleshift"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_prediction_is_explicit_failure(mode, bad):
    gt, valid = _scene()
    pred = gt.clone()
    pred[0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match="non-finite prediction"):
        align(pred, gt, valid, mode, "depth")
