"""The point of tce: unlike t-delta and opw, a constant prediction must not win."""
import torch

from sokkanaem.metrics import temporal_metrics


def _pan_sequence(T=4, H=128, W=128):  # RAFT needs >=128px (corr pyramid)
    """Camera panning right over a scene whose depth varies horizontally, so
    each pixel's depth genuinely changes frame to frame."""
    xx = torch.linspace(0, 1, W * 2).view(1, W * 2).expand(H, W * 2)
    frames, gt = [], []
    for t in range(T):
        s = slice(t * 4, t * 4 + W)
        strip = xx[:, s]
        frames.append(torch.stack([strip, strip * 0.5, 1 - strip]))  # textured
        gt.append((strip * 8 + 2).unsqueeze(0))                      # 2..10 m
    return torch.stack(frames), torch.stack(gt)


def test_constant_prediction_loses_on_tce():
    frames, gt = _pan_sequence()
    const = torch.full_like(gt, gt.median())

    tdelta = (const[1:] - const[:-1]).abs().mean().item()
    m = temporal_metrics(frames, const, gt)

    # both legacy metrics are fooled: a constant is perfectly "stable"
    assert tdelta == 0.0
    assert m["opw"] < 1e-6
    # tce is not: it charges the constant for GT's real inter-frame change
    assert m["tce"] > 0.01, m


def test_perfect_prediction_scores_near_zero():
    frames, gt = _pan_sequence()
    m = temporal_metrics(frames, gt.clone(), gt)
    assert m["tce"] < m_const_tce(frames, gt), "GT must beat a constant"
    assert m["tce"] < 0.05, m


def m_const_tce(frames, gt):
    const = torch.full_like(gt, gt.median())
    return temporal_metrics(frames, const, gt)["tce"]


def test_validity_mask_excludes_invalid_gt():
    frames, gt = _pan_sequence()
    valid = torch.ones_like(gt)
    valid[:, :, :, :32] = 0
    m = temporal_metrics(frames, gt.clone(), gt, valid)
    assert m["tce"] >= 0.0 and m["opw"] >= 0.0


def test_clip_with_no_valid_gt_is_skipped():
    """A real-sensor clip can have zero valid GT pixels. t-delta/OPW/TCE are
    measured over ALL pixels, so if such a clip is scored its nan scale poisons
    the pooled sums for the whole dataset (observed on the Bonn/TUM holdout)."""
    from sokkanaem.metrics import clip_scores
    frames, gt = _pan_sequence()
    assert clip_scores(frames, gt.clone(), gt, torch.zeros_like(gt)) is None
    assert clip_scores(frames, gt.clone(), gt, torch.ones_like(gt)) is not None
