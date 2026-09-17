"""Development-only GT-flat retention and bounded refresh-state carry."""
import torch
import torch.nn.functional as F
from sokkanaem.sharpness import norm_disp, grad_mag, boundary_band, _masked_quantile, _dilate
from scripts.qquality_study import MotionStream


@torch.no_grad()
def flat_mask(gt, valid):
    valid = valid.bool() & torch.isfinite(gt) & (gt > 0) & (gt < 150)
    gradient, defined = grad_mag(norm_disp(gt, valid), valid)
    enough = defined.flatten(1).sum(1).reshape(-1, 1, 1, 1) >= 10
    threshold = _masked_quantile(gradient, defined.bool(), .5)
    edge = _dilate(boundary_band(gt, valid, dilate=3), 3)
    mask = (gradient <= threshold) & defined.bool() & enough & ~edge
    return mask, valid


def flat_excess_loss(pred, teacher, valid, mask):
    """Only penalize excess normalized-disparity roughness over Q0 in GT-flat areas."""
    gp, _ = grad_mag(norm_disp(pred, valid), valid)
    gt, _ = grad_mag(norm_disp(teacher.detach(), valid), valid)
    n = mask.flatten(1).sum(1)
    loss = ((gp-gt).relu()*mask).flatten(1).sum(1)/n.clamp_min(1)
    keep = n >= 10
    return (loss*keep).sum()/keep.sum().clamp_min(1)


def flat_plane_loss(pred, teacher, valid, mask, win=9):
    """Penalize curvature the reference does not have, inside GT-flat areas, leaving slope free.

    `flat_excess_loss` penalizes any normalized-disparity gradient above the reference's, which
    on a slanted plane removes the slope itself: with Kinect labels quantized flat, and even on
    synthetic ones, it cost the MambaVision probe 0.139 -> 0.24 mean |log error| in Date04's
    2-3 m band (SOKKANAEM_MV_DECODER_PROBE_20260915.md I.5, I.10).

    A least-squares plane fitted over a symmetric window evaluates at the window centre to the
    window mean, so `x - boxmean(x)` IS that plane's residual at the centre: any ramp gives
    exactly zero, a bump does not. Only the prediction's residual beyond the reference's own is
    paid for. Windows are truncated at the image border (count_include_pad=False), where a ramp
    leaves a small residual on both maps and largely cancels in the difference.
    """
    shape = (-1, 1) + tuple(pred.shape[-2:])
    p, t, v = (x.reshape(shape) for x in (pred, teacher, valid))
    pd, td = norm_disp(p, v), norm_disp(t.detach(), v)
    box = lambda x: F.avg_pool2d(x, win, 1, win // 2, count_include_pad=False)
    excess = ((pd - box(pd)).abs() - (td - box(td)).abs()).clamp(min=0)
    m = mask.reshape(shape) if mask.shape != shape else mask
    n = m.flatten(1).sum(1)
    loss = (excess * m).flatten(1).sum(1) / n.clamp_min(1)
    keep = n >= 10
    return (loss * keep).sum() / keep.sum().clamp_min(1)


def flat_band_loss(pred, teacher, valid, mask, win_lo=5, win_hi=17):
    """Penalize excess disparity energy in the win_lo..win_hi pixel band only (a
    difference-of-box bandpass: box(x, win_lo) - box(x, win_hi)), leaving both finer detail
    (boundary sharpness, < win_lo) and coarser structure (slope, > win_hi) untouched.

    SOKKANAEM_MV_DECODER_PROBE_20260915.md I.20 found the 592px probe's flat-TV excess
    concentrated in the 4-16px band and disappearing by 32px. `flat_plane_loss`'s single-window
    high-pass (`x - box(x, win)`) keeps ALL finer content too (including whatever content is
    driving boundary F1), which is why raising its weight erodes F1 alongside flat TV (I.19,
    I.26). This term is a proper bandpass and should not have that coupling.
    """
    shape = (-1, 1) + tuple(pred.shape[-2:])
    p, t, v = (x.reshape(shape) for x in (pred, teacher, valid))
    pd, td = norm_disp(p, v), norm_disp(t.detach(), v)
    box = lambda x, w: F.avg_pool2d(x, w, 1, w // 2, count_include_pad=False)
    band = lambda x: box(x, win_lo) - box(x, win_hi)
    excess = (band(pd).abs() - band(td).abs()).clamp(min=0)
    m = mask.reshape(shape) if mask.shape != shape else mask
    n = m.flatten(1).sum(1)
    loss = (excess * m).flatten(1).sum(1) / n.clamp_min(1)
    keep = n >= 10
    return (loss * keep).sum() / keep.sum().clamp_min(1)


def refresh_memory(fresh, old, frame, previous, carry, hard_reset=False):
    """Refresh features exactly, retain .9h only at low RGB-change locations.

    No motion alignment/occlusion guarantee: this is a conservative pixel-grid
    heuristic. Never changes fresh Q0 features, pooled calibration or depth.
    """
    if not carry or old is None or hard_reset or old["h"].shape != fresh["h"].shape:
        return fresh, 0
    gh, gw = fresh["grid"]
    size = (gh*14, gw*14)
    a = F.interpolate(frame, size, mode="bilinear", align_corners=False)
    b = F.interpolate(previous, size, mode="bilinear", align_corners=False)
    error = F.avg_pool2d((a-b).square().mean(1, keepdim=True), 14).reshape(-1)
    keep = (error <= .002).to(old["h"].dtype)
    h = old["h"]*.9*keep[:, None, None]
    return {**fresh, "h": h}, int(keep.sum())


class PersistentStream(MotionStream):
    def __init__(self, exact, adapter, carry=False):
        super().__init__(exact, adapter, refresh_every=2)
        self.carry = carry

    @torch.no_grad()
    def step(self, frame, state=None):
        previous = state
        n = 0 if state is None else state["frame_number"]
        pred, state, info = super().step(frame, state)
        retained = 0
        if info["full_refresh"] and previous is not None:
            state["features"], retained = refresh_memory(state["features"], previous["features"], frame,
                                                        previous["previous"], self.carry, hard_reset=n % 32 == 0)
        state["frame_number"] = n+1
        info["retained_h_locations"] = retained
        return pred, state, info
