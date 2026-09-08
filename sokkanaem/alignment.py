"""One alignment implementation, shared by every model in the ACC comparison.

PLAN_ACC A2. The old arrangement had three: scripts/eval.py fitted scale+shift
with numpy inside `eval_once`, scripts/eval_baseline_da2.py fitted its own, and
scripts/viz_qualitative.py a third. They agreed by inspection rather than by
construction, and all three hid the same defect — a 2-DOF disparity fit can put
valid pixels at or below zero disparity, and `1/clamp(disp, 1e-3)` turns an
undefined depth into a large-but-finite 1000 m error that the mean quietly
absorbs. DA V2 Small's mean AbsRel of 0.2256 against a clip median of 0.0967 is
consistent with that failure mode (PLAN_ACC §1.2). It reflects a model–gauge
interaction, not a gauge-independent measure of model quality.

So alignment returns the failure alongside the depth, and the caller counts it.

Gauges, all fitted per clip over the clip's valid pixels:

    median      1 DOF in DEPTH space, s = median(gt)/median(pred). What every
                metric-output SOKKANAEM numbers historically use. It removes
                absolute-scale error; this is NOT uncalibrated metric accuracy.
    none        no GT fit, only for a model whose output is metric depth.
    scale       1 DOF in disparity space, y ~= s*x; auxiliary diagnostic.
    scaleshift  2 DOF in disparity space, y ~= s*x + b. The MiDaS protocol
                relative-depth baselines are designed for, and the primary
                gauge for relative-shape comparison in the paper protocol.
                Its least-squares objective is not depth AbsRel.
"""
import torch

MODES = ("none", "median", "scale", "scaleshift")
LEGACY_MODES = ("median", "scale", "scaleshift")

# Depth floor for the disparity round-trip. 1e-3 m matches what eval.py and the
# baseline scripts already used, so numbers stay comparable to the existing
# tables; the point of this module is not to change the clamp but to report
# what it swallowed.
EPS = 1e-3


def _fit(x, y, dof):
    """Least squares y ~= s*x (+ b) over 1-D x, y. Returns (s, b) as floats.

    Closed form rather than lstsq: it is the same fit for a [x, 1] design
    matrix, and it keeps the whole thing on-device in float64 instead of
    round-tripping a few million pixels through numpy per clip."""
    x, y = x.double(), y.double()
    if dof == 1:
        s = (x * y).sum() / (x * x).sum().clamp(min=1e-12)
        return s.item(), 0.0
    mx, my = x.mean(), y.mean()
    s = ((x - mx) * (y - my)).sum() / ((x - mx) ** 2).sum().clamp(min=1e-12)
    return s.item(), (my - s * mx).item()


def align(pred, gt, valid, mode="scaleshift", space="depth", per_frame=False):
    """Align `pred` to `gt`'s gauge, per clip or per frame.

    per_frame fits the gauge on each frame independently, which is the other
    half of PLAN_ACC §1.3's instruction to report both. One fit per clip and
    one per frame answer different questions and a long-stream number quoted
    from only the first is unreadable: measured here, a STATELESS 343M
    DPT-Large loses 116% going from 8-frame to 256-frame clips under the
    clip-wide fit. A per-frame fit removes time-varying global scale/shift,
    which can itself be real model inconsistency. Different clip-length sets
    also contain different frames; this comparison alone does not identify
    recurrent drift. Both gauges and a matched-frame experiment are needed.

    A frame with no valid GT keeps the clip-level fit rather than becoming
    nan -- its pixels are masked out of every accuracy metric anyway, but
    t-delta and OPW are measured over all of them.
    """
    clip = _align_clip(pred, gt, valid, mode, space)
    if not per_frame or clip is None or mode == "none":
        return clip
    depth, _ = clip
    out, negs, failed = depth.clone(), [], 0
    for t in range(pred.shape[0]):
        a = _align_clip(pred[t:t + 1], gt[t:t + 1], valid[t:t + 1], mode, space)
        if a is None:
            continue
        out[t:t + 1], i = a
        negs.append(i["neg_frac"])
        failed += int(i["failed"])
    return out, {"mode": f"{mode}/frame", "s": float("nan"), "b": float("nan"),
                 "neg_frac": sum(negs) / max(len(negs), 1), "failed": failed > 0,
                 "failed_frames": failed}


def _align_clip(pred, gt, valid, mode="scaleshift", space="depth"):
    """One gauge fitted over every valid pixel of the whole tensor.
    Returns (depth, info) or None.

    pred, gt, valid: same shape, any leading dims. gt is metric depth in
    metres with 0 = invalid; `space` says what pred is:

        "depth"      metres-like, larger = farther (ours, ZoeDepth, VDA-metric)
        "disparity"  relative model output, larger = nearer (DPT, DA V1/V2/V3)

    None when the clip has no valid pixel at all — real depth sensors drop
    whole frames, and a clip with no median cannot be aligned by any gauge.

    info carries the fit and what it broke:
        s, b        fitted parameters (b = 0 for the 1-DOF gauges)
        neg_frac    fraction of nonpositive predictions (none/median), or
                    nonpositive fitted disparities (scale/scaleshift)
        failed      neg_frac > 0. Count these clips explicitly and retain their
                    errors in pooled scores; never silently drop failures.
    """
    assert mode in MODES, mode
    assert space in ("depth", "disparity"), space
    v = valid.bool()
    if not bool(v.any()):
        return None

    if not bool(torch.isfinite(pred).all()):
        raise ValueError("non-finite prediction: cannot report a finite accuracy score")
    if mode == "none":
        if space != "depth":
            raise ValueError("align=none requires metric depth, not relative disparity")
        neg = float((pred[v] <= 0).float().mean().item())
        return pred, {"mode": mode, "s": 1.0, "b": 0.0,
                      "neg_frac": neg, "failed": neg > 0}

    if mode == "median":
        depth = pred if space == "depth" else 1.0 / pred.clamp(min=EPS)
        s = (gt[v].median() / depth[v].median().clamp(min=EPS)).item()
        neg = float((pred[v] <= 0).float().mean().item())
        return depth * s, {"mode": mode, "s": s, "b": 0.0,
                           "neg_frac": neg, "failed": neg > 0}

    x = pred if space == "disparity" else 1.0 / pred.clamp(min=EPS)
    y = 1.0 / gt.clamp(min=EPS)
    s, b = _fit(x[v], y[v], 1 if mode == "scale" else 2)
    disp = s * x + b
    neg = float((disp[v] <= 0).float().mean().item())
    return 1.0 / disp.clamp(min=EPS), {
        "mode": mode, "s": s, "b": b, "neg_frac": neg, "failed": neg > 0}


def _selfcheck():
    torch.manual_seed(0)
    gt = torch.rand(4, 1, 16, 16) * 4 + 0.5
    valid = torch.ones_like(gt)

    # gauge invariance: each mode must undo exactly the freedom it fits, so a
    # prediction that IS the ground truth under that freedom aligns back to it.
    for mode, space, pred in (
            ("median", "depth", gt * 3.7),
            ("scale", "depth", gt * 3.7),
            ("scale", "disparity", 1.0 / gt * 0.21),
            ("scaleshift", "disparity", 1.0 / gt * 0.21 + 0.4)):
        out, info = align(pred, gt, valid, mode, space)
        err = (out - gt).abs().max().item()
        assert err < 1e-3, (mode, space, err, info)
        assert not info["failed"], info

    # per-frame is a strictly tighter fit than one gauge for the whole clip:
    # a prediction whose scale wanders frame to frame is recovered exactly by
    # the first and not at all by the second, which is the whole reason G2
    # needs both numbers
    drift = torch.stack([gt[t] * (1.0 + 0.4 * t) for t in range(gt.shape[0])])
    a, _ = align(drift, gt, valid, "median", "depth")
    b, ib = align(drift, gt, valid, "median", "depth", per_frame=True)
    assert (b - gt).abs().max() < 1e-4, ib
    assert (a - gt).abs().max() > 0.1, "clip-wide fit must NOT absorb the drift"

    # a shift the fit must undo is not a failure; a fit that has to send valid
    # pixels through zero disparity IS one, and must say so rather than clamp
    # it into a finite error
    far = gt.clone()
    far[0] = 900.0                       # near-zero GT disparity in one frame
    pred = 1.0 / far + 0.5               # a shift the 2-DOF fit removes exactly
    out, info = align(pred, far, valid, "scaleshift", "disparity")
    assert not info["failed"], info
    out, info = align(pred, far, valid, "scale", "disparity")
    assert info["neg_frac"] == 0.0, info  # 1 DOF cannot 0-cross

    # the real failure mode: a 2-DOF fit whose best line dips below zero over
    # the x range that is actually present. GT disparity 1,1,10 against a
    # prediction of 0,1,2 fits s=4.5, b=-0.5 -- the first pixel lands at
    # disparity -0.5, which is not a depth at all. Clamping it reports 1000 m.
    g = torch.tensor([1.0, 1.0, 0.1]).view(3, 1, 1, 1)      # disparity 1,1,10
    p = torch.tensor([0.0, 1.0, 2.0]).view(3, 1, 1, 1)
    out, info = align(p, g, torch.ones_like(g), "scaleshift", "disparity")
    assert info["failed"] and abs(info["neg_frac"] - 1 / 3) < 1e-6, info
    assert abs(info["s"] - 4.5) < 1e-6 and abs(info["b"] + 0.5) < 1e-6, info
    assert abs(out[0].item() - 1.0 / EPS) < 1e-3, out  # the clamp's fiction

    # no valid pixel -> no gauge
    assert align(gt, gt, torch.zeros_like(gt), "median") is None
    print("alignment ok")


if __name__ == "__main__":
    _selfcheck()
