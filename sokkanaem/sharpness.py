"""Sharpness metrics: what AbsRel cannot see.

`scripts/sharp_metric.py` already reported two numbers -- edge-band AbsRel and
the boundary gradient ratio. The ratio alone is buyable with noise: add
high-frequency garbage everywhere and the prediction's gradients grow, so the
ratio rises while the depth map gets worse. Three metrics here close that
loophole, all on per-frame scale-shift-normalized disparity so no alignment
rule can flatter them (PLAN.md sections 3.2 and 5):

  boundary P/R/F1   GT boundaries are the pixels whose normalized-disparity
                    gradient sits above the GT's own q-quantile; the predicted
                    set is thresholded at the SAME absolute value, so a blurred
                    prediction simply fails to reach it and loses recall, while
                    a noisy one detects everywhere and loses precision.
                    Matching allows `tol` px of slack. Averaged over q.

  flat TV           mean |grad| of the prediction inside GT-flat regions, and
                    its ratio to the GT's. This is the noise detector: it is
                    the term that goes up when the gradient ratio is bought
                    rather than earned.

  overshoot         fraction of boundary-band pixels whose value leaves the
                    local GT range by more than `margin` of that range --
                    ringing, the other way a sharp-looking map can be wrong.

Every function takes depth in metres, (N,1,H,W), with a validity mask, and is
invariant to a scale+shift of the prediction's disparity by construction.
"""
import torch
import torch.nn.functional as F


def _masked_quantile(x, keep, q):
    """Per-frame quantile of x over `keep` pixels. x/keep (N,1,H,W), q a float
    or a tuple; returns (N,1,1,1) or (len(q),N,1,1,1).

    Batched on purpose: the frame-by-frame version cost 235 ms per training
    step in `boundary_location_loss` -- one `torch.quantile` launch and one
    host sync per frame -- against 2.8 ms for the whole of
    `edge_weighted_loss`. Same numbers, one kernel."""
    v = torch.where(keep.bool(), x, torch.nan).flatten(1).float()
    qq = torch.as_tensor(q, dtype=torch.float32, device=x.device)
    out = torch.nanquantile(v, qq, dim=-1)          # (…, N)
    return out.reshape(out.shape[:-1] + (x.shape[0], 1, 1, 1)).to(x.dtype)


def norm_field(x, valid):
    """Per-frame median-centre, mean-absolute-deviation scale, over valid
    pixels. x/valid (N,1,H,W).

    Per frame, not per batch: a batch holding one 0.5-129 m scene would
    otherwise set the scale for a 1.5-4 m one and flatten it to nothing."""
    v = valid.bool()
    t = torch.nanmedian(torch.where(v, x, torch.nan).flatten(1), dim=1
                        ).values.reshape(-1, 1, 1, 1)
    t = torch.nan_to_num(t)
    n = v.flatten(1).sum(1).reshape(-1, 1, 1, 1)
    s = (((x - t).abs() * v).flatten(1).sum(1).reshape(-1, 1, 1, 1)
         / n.clamp(min=1)).clamp(min=1e-6)
    return torch.where(n > 0, (x - t) / s, torch.zeros_like(x))


def norm_disp(depth, valid):
    """Per-frame scale-shift normalized disparity (MiDaS)."""
    return norm_field(1.0 / depth.clamp(min=1e-3), valid)


def grad_mag(x, valid):
    """|grad| and where it is defined, forward differences, same shape as x.

    Both endpoints of a difference must be valid. Without that, every sensor
    hole contributes a gradient the size of the hole's depth, the top decile
    of the GT gradient becomes the set of hole borders rather than the set of
    object boundaries, and the ratio below reads ~0 for any model."""
    vy = valid[..., 1:, :] * valid[..., :-1, :]
    vx = valid[..., :, 1:] * valid[..., :, :-1]
    gy = F.pad((x[..., 1:, :] - x[..., :-1, :]).abs() * vy, (0, 0, 0, 1))
    gx = F.pad((x[..., :, 1:] - x[..., :, :-1]).abs() * vx, (0, 1))
    defined = F.pad(vy, (0, 0, 0, 1)) * F.pad(vx, (0, 1))
    return gy + gx, defined


def boundary_band(gt, valid, decile=0.9, dilate=3):
    """Top-decile GT log-depth gradient, dilated. (N,1,H,W) bool."""
    g, defined = grad_mag(gt.clamp(min=1e-3).log(), valid)
    g = F.max_pool2d(g, dilate, stride=1, padding=dilate // 2) * defined
    d = defined.bool()
    thr = _masked_quantile(g, d, decile)
    enough = d.flatten(1).sum(1).reshape(-1, 1, 1, 1) >= 10
    # a scene flat enough that the q-quantile is 0 would otherwise call every
    # pixel a boundary, and every metric below then reads 1.0
    return (g >= thr) & (g > 0) & d & enough


def _dilate(mask, tol):
    """Grow a bool mask by `tol` px (square structuring element)."""
    k = 2 * tol + 1
    return F.max_pool2d(mask.float(), k, stride=1, padding=tol) > 0


def grad_ratio(pred, gt, valid, band=None):
    """mean |grad(norm_disp(pred))| / mean |grad(norm_disp(gt))| in the band.

    1.0 = varies as fast as the truth across a boundary, 0.5 = twice as
    smooth. Above 1.0 is not credit: check flat_tv and precision."""
    band = boundary_band(gt, valid) if band is None else band
    if not bool(band.any()):
        return float("nan")
    gp, _ = grad_mag(norm_disp(pred, valid), valid)
    gg, _ = grad_mag(norm_disp(gt, valid), valid)
    return float(gp[band].mean() / gg[band].mean().clamp(min=1e-6))


def boundary_prf(pred, gt, valid, quantiles=(0.90, 0.95, 0.99), tol=2):
    """Multi-threshold boundary precision/recall/F1, averaged over quantiles.

    The threshold comes from the GT's gradient distribution and is applied to
    both maps, which is what makes the metric one-sided in the right way:
    blur cannot reach the threshold (recall falls), noise reaches it
    everywhere (precision falls)."""
    pd, gd = norm_disp(pred, valid), norm_disp(gt, valid)
    gp, defined = grad_mag(pd, valid)
    gg, _ = grad_mag(gd, valid)
    d = defined.bool()
    enough = d.flatten(1).sum(1).reshape(-1, 1, 1, 1) >= 10
    thr = _masked_quantile(gg, d, tuple(quantiles))        # (Q,N,1,1,1)
    acc = {"precision": [], "recall": [], "f1": []}
    for t in thr:                                          # one pass per q
        bg = (gg >= t) & (gg > 0) & d & enough
        bp = (gp >= t) & (gp > 0) & d & enough
        ng = bg.flatten(1).sum(1).float()
        np_ = bp.flatten(1).sum(1).float()
        hit_p = (bp & _dilate(bg, tol)).flatten(1).sum(1).float()
        hit_r = (bg & _dilate(bp, tol)).flatten(1).sum(1).float()
        keep = ng > 0                    # no GT boundary at this q: skip frame
        if not bool(keep.any()):
            continue
        p = torch.where(np_ > 0, hit_p / np_.clamp(min=1), torch.zeros_like(np_))
        r = hit_r / ng.clamp(min=1)
        f1 = torch.where(p + r > 0, 2 * p * r / (p + r).clamp(min=1e-9),
                         torch.zeros_like(p))
        for k, x in (("precision", p), ("recall", r), ("f1", f1)):
            acc[k].extend(x[keep].tolist())
    return {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}


def flat_tv(pred, gt, valid, quantile=0.5, dilate=3):
    """Total variation of the prediction inside GT-flat regions, and the ratio
    to the GT's own. Flat = GT gradient below its `quantile`, excluding the
    dilated boundary band so a smeared edge is not counted as flat noise.

    Ratio near 1 means the prediction is as quiet as the truth; >> 1 means
    high-frequency noise, which is how a bought gradient ratio shows up. On
    Kinect GT a flat region is quantized to a constant, so its own TV is near
    zero and the ratio reads in the tens (77 on Bonn) -- read `flat_tv`
    against `flat_tv_gt` there, and the ratio only where GT is continuous."""
    pd, gd = norm_disp(pred, valid), norm_disp(gt, valid)
    gp, defined = grad_mag(pd, valid)
    gg, _ = grad_mag(gd, valid)
    d = defined.bool()
    enough = d.flatten(1).sum(1).reshape(-1, 1, 1, 1) >= 10
    thr = _masked_quantile(gg, d, quantile)
    near_edge = _dilate(boundary_band(gt, valid, dilate=dilate), dilate)
    flat = (gg <= thr) & d & enough & ~near_edge
    n = flat.flatten(1).sum(1).float()
    keep = n >= 10
    if not bool(keep.any()):
        return {k: float("nan")
                for k in ("flat_tv", "flat_tv_gt", "flat_tv_ratio")}
    per = lambda x: ((x * flat).flatten(1).sum(1) / n.clamp(min=1))[keep]
    tv, gtv = per(gp), per(gg)
    return {"flat_tv": float(tv.mean()),
            "flat_tv_gt": float(gtv.mean()),
            "flat_tv_ratio": float((tv / gtv.clamp(min=1e-6)).mean())}


def overshoot(pred, gt, valid, band=None, win=15, margin=0.05):
    """Fraction of boundary-band pixels leaving the local GT range by more
    than `margin` x that range -- ringing next to an edge.

    This metric exists so that fixing blur cannot be reported as a win when
    it was paid for with overshoot, so it must read 0 for blur: `win` has to
    exceed the smear radius, or a blurred edge's value -- correct, just in the
    wrong place -- falls outside a window holding one side of the edge only.
    Measured on a 96px synthetic edge: a 9px box blur scores 0.35 at win=7,
    0.15 at win=11 and 0.00 at win=15, while a 1.5x unsharp mask still scores
    0.19."""
    band = boundary_band(gt, valid) if band is None else band
    if not bool(band.any()):
        return float("nan")
    pd, gd = norm_disp(pred, valid), norm_disp(gt, valid)
    v = valid.bool()
    # invalid pixels must not set the local range: push them to -inf on
    # whichever side is being maxed over
    big = torch.full_like(gd, 1e4)   # finite: see overshoot_loss's note
    hi = F.max_pool2d(torch.where(v, gd, -big), win, stride=1,
                      padding=win // 2)
    lo = -F.max_pool2d(torch.where(v, -gd, -big), win, stride=1,
                       padding=win // 2)
    rng = (hi - lo).clamp(min=1e-6)
    out = (pd - hi).clamp(min=0) + (lo - pd).clamp(min=0)
    return float(((out / rng) > margin)[band & v].float().mean())


def sharpness_scores(pred, gt, valid, band=None):
    """Every sharpness number for one batch of frames, as one dict.

    `pred`, `gt`, `valid`: (N,1,H,W), depth in metres. Callers hold the
    alignment rule; these metrics do not need one."""
    band = boundary_band(gt, valid) if band is None else band
    rel = (pred - gt).abs() / gt.clamp(min=1e-3)
    out = {"absrel_edge": float(rel[band].mean()) if bool(band.any())
           else float("nan"),
           "grad_ratio": grad_ratio(pred, gt, valid, band),
           "overshoot": overshoot(pred, gt, valid, band)}
    out.update({f"boundary_{k}": v
                for k, v in boundary_prf(pred, gt, valid).items()})
    out.update(flat_tv(pred, gt, valid))
    return out
