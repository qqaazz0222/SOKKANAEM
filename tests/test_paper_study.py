"""Extra study equations and interventions match the frozen scorer/model."""
import numpy as np
from PIL import Image
import pytest
import torch

from sokkanaem import SOKKANAEM
from sokkanaem import metrics
from sokkanaem.study import (LinearConvCounter, grid_from_flow, native_predict,
    original_views, output_hold, regions_from_flow, temporal_from_grid)


def test_linear_conv_counter_counts_executed_shapes():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.Flatten(), torch.nn.Linear(16, 2))
    counter = LinearConvCounter(model)
    model(torch.ones(1, 3, 4, 4))
    assert counter.total == 4 * 2 * 2 * 3 * 3 * 3 + 16 * 2
    counter.close()
    model(torch.ones(1, 3, 4, 4))
    assert counter.total == 464


def test_patch_output_hold_has_no_hidden_state_assumption():
    current, previous = torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)
    mask = torch.tensor([[1., 0., 0., 0.]])
    out = output_hold(current, previous, mask, 2)
    assert out[..., :2, :2].sum() == 4 and out.sum() == 4


def test_precomputed_flow_matches_frozen_temporal_and_regions(monkeypatch):
    from scripts.eval_acc import region_masks
    torch.manual_seed(4)
    frames = torch.rand(3, 3, 16, 16)
    flow = torch.randn(2, 2, 16, 16) * 2
    gt = torch.rand(3, 1, 16, 16) * 4 + .5
    valid = (torch.rand_like(gt) > .1).float()
    pred = gt * 1.2 + .1
    monkeypatch.setattr(metrics, "_flow", lambda *a, **k: flow)
    grid, inb = grid_from_flow(flow)
    reference_grid, reference_inb = metrics.warp_grids(frames)
    assert torch.equal(grid, reference_grid) and torch.equal(inb, reference_inb)
    expected = metrics.temporal_metrics(frames, pred, gt, valid, pooled=True)
    actual = temporal_from_grid(pred, gt, valid, grid, inb)
    assert actual == pytest.approx(expected)
    expected_regions = region_masks(frames, gt, valid)
    actual_regions = regions_from_flow(gt, valid, flow)
    for name in expected_regions:
        assert torch.equal(expected_regions[name], actual_regions[name]), name


def test_forced_replay_and_reset_match_model_contract():
    torch.manual_seed(5)
    model = SOKKANAEM(dim=32, depth=2, d_state=4, spatial_cache=True,
                      temporal_cache=True, keyframe_every=3).eval()
    frames = torch.rand(4, 3, 32, 32)
    frames[1:] = frames[0]
    expected, masks, mac = native_predict(model, frames, {}, count=True)
    actual, replay_masks, _ = native_predict(model, frames, {"forced": True}, masks)
    assert torch.equal(masks, replay_masks)
    assert torch.allclose(expected, actual) and mac > 0
    reset, all_masks, _ = native_predict(model, frames, {"reset": True})
    assert bool(all_masks.all())
    assert torch.allclose(reset, expected[:1].expand_as(reset), atol=1e-5)


def test_original_view_preserves_source_resolution_and_scoring_roi(tmp_path):
    w, h = 640, 480
    values = np.tile(np.arange(w) * 255 / w, (h, 1)).astype(np.uint8)
    path = tmp_path / "rgb.png"
    Image.fromarray(np.stack([values] * 3, -1)).save(path)
    view = original_views([[str(path), "unused"]])[0]
    assert view.width >= 480 and view.height >= 480
    from sokkanaem.data import ClipDataset
    reference = ClipDataset([], 1000, size=256)._rgb(str(path))
    reduced = torch.from_numpy(np.asarray(view.resize((256, 256), Image.Resampling.BILINEAR)).copy()).permute(2, 0, 1) / 255
    assert (reference - reduced).abs().max() < 2 / 255


def test_report_uses_equal_source_not_equal_clip_weighting():
    from scripts.paper_study_report import aggregate
    def row(source, error):
        sums = {"rel_sum": error, "sq_sum": error**2, "d1_sum": 0., "px": 1.,
                "td_sum": 0., "td_px": 1., "opw_sum": 0., "tce_sum": 0., "warp_px": 1.}
        return {"source": source, "models": {"m": {"params": 1, "effective_size": [16, 16],
            "future_frames": False, "gauges": {"none": {"scores": {"absrel": error, "_pooled": sums},
                "failed": False, "catastrophic": error > 1}}}}}
    report = aggregate([row("a", 3.), row("b", 0.), row("b", 0.)])
    values = report[("m", "none")]["balanced"]
    assert values["absrel"] == 1.5  # clip-weighted would incorrectly be 1.0
    assert values["n"] == 3 and values["catastrophic"] == 1


def test_frame_curve_does_not_turn_no_gt_into_perfect_prediction():
    from scripts.paper_study_report import frame_curve
    row = {"source": "a", "models": {"m": {"gauges": {"none": {
        "frames": {"rel_sum": [2., 0.], "d1_sum": [0., 0.], "px": [2., 0.]}}}}}}
    assert frame_curve([row], "m", "none") == [1., None]
