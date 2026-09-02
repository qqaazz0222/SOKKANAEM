"""Training losses (IDEA.md §3.3), all masked by depth validity."""
import torch
import torch.nn.functional as F


def si_log_loss(pred, gt, valid):
    d = (torch.log(pred.clamp(min=1e-6)) - torch.log(gt.clamp(min=1e-6))) * valid
    n = valid.sum().clamp(min=1)
    return (d ** 2).sum() / n - 0.5 * (d.sum() / n) ** 2


def grad_loss(pred, gt, valid):
    def gy(x):
        return x[..., 1:, :] - x[..., :-1, :]

    def gx(x):
        return x[..., :, 1:] - x[..., :, :-1]

    vy = valid[..., 1:, :] * valid[..., :-1, :]
    vx = valid[..., :, 1:] * valid[..., :, :-1]
    return (((gy(pred) - gy(gt)).abs() * vy).sum() / vy.sum().clamp(min=1)
            + ((gx(pred) - gx(gt)).abs() * vx).sum() / vx.sum().clamp(min=1))


def normal_loss(pred, gt, valid):
    """1 - cosine similarity between depth-gradient pseudo surface normals.
    No camera intrinsics are tracked anywhere in this codebase (depth is
    just per-pixel meters), so normals live in image-plane coordinates:
    n ∝ (-dz/dx, -dz/dy, 1) rather than true unprojected 3D normals —
    still penalizes cross-object depth-slope mismatches at edges."""
    def normals(d):
        dzdx = (d[..., :, 1:] - d[..., :, :-1])[..., :-1, :]
        dzdy = (d[..., 1:, :] - d[..., :-1, :])[..., :, :-1]
        n = torch.stack([-dzdx, -dzdy, torch.ones_like(dzdx)], dim=-1)
        return F.normalize(n, dim=-1, eps=1e-6)

    # each normal at (i,j) consumes depth at (i,j), (i,j+1), (i+1,j) — all
    # three must be valid or the slope is garbage (sky sentinels in vkitti2/
    # tartanair). Match grad_loss's both-endpoints masking rigor.
    v = valid[..., :-1, :-1] * valid[..., :-1, 1:] * valid[..., 1:, :-1]
    cos = (normals(pred) * normals(gt)).sum(-1)
    return ((1 - cos) * v).sum() / v.sum().clamp(min=1)


def temporal_loss(depths, masks, patch=16):
    """Static regions must keep identical depth across frames."""
    B, T = depths.shape[:2]
    if T < 2:
        return depths.sum() * 0
    static = 1 - masks[:, 1:]                                # (B, T-1, N)
    gh = depths.shape[-2] // patch
    m = static.reshape(B, T - 1, gh, -1)
    m = F.interpolate(m.reshape(B * (T - 1), 1, gh, -1).float(),
                      scale_factor=patch, mode="nearest")
    diff = (depths[:, 1:] - depths[:, :-1]).reshape(B * (T - 1), 1, *depths.shape[-2:])
    return (diff.abs() * m).sum() / m.sum().clamp(min=1)


def warp_residual_loss(frames, pred, gt, valid, min_px=128):
    """Training-time TCE: match the flow-warped residual of the prediction to
    GT's own warped residual (REPORT §4.20b — bin CE buys accuracy by paying
    8-12% temporal stability, and `temporal_loss` cannot buy it back because it
    only sees *static* patches and a constant output is its global optimum).

    Everything is in log-depth, so the term is invariant to a per-clip scale:
    only the frame-to-frame *change* along a flow track is supervised.

    frames (B,T,3,H,W) in [0,1]; pred/gt/valid (B,T,1,H,W). The RAFT flow is
    computed under no_grad and reused as a fixed correspondence, exactly like
    the eval metric, so the gradient flows through grid_sample only.
    """
    from .metrics import warp, warp_grids

    B, T = pred.shape[:2]
    if T < 2 or min(pred.shape[-2:]) < min_px:
        return pred.sum() * 0                    # RAFT needs >=128px
    lp, lg = pred.clamp(min=1e-3).log(), gt.clamp(min=1e-3).log()
    total, denom = pred.sum() * 0, 0.0
    for b in range(B):
        grid, inb = warp_grids(frames[b])
        m = inb * valid[b, 1:] * warp(valid[b, :-1], grid)
        if m.sum() < 1:
            continue
        rp = warp(lp[b, :-1], grid) - lp[b, 1:]
        rg = warp(lg[b, :-1], grid) - lg[b, 1:]
        total = total + ((rp - rg).abs() * m).sum() / m.sum()
        denom += 1
    return total / max(denom, 1.0)


def edge_weighted_loss(pred, gt, valid, dilate=3):
    """Log-depth L1 weighted towards depth discontinuities.

    Foreground objects are exactly the pixels with a large GT depth gradient,
    and a 32px constant-per-block oracle beats this model there (§4.17): the
    global si-log average is dominated by the large smooth background, so the
    boundary band contributes almost nothing to the gradient. Weight is the
    max-pooled GT log-depth gradient magnitude, normalized to mean 1 over the
    valid pixels so the term's scale does not depend on the scene's depth range.
    """
    lp, lg = pred.clamp(min=1e-3).log(), gt.clamp(min=1e-3).log()
    f = lambda x: x.reshape(-1, 1, *x.shape[-2:])
    lp, lg, v = f(lp), f(lg), f(valid)
    gy = (lg[..., 1:, :] - lg[..., :-1, :]).abs() * (v[..., 1:, :] * v[..., :-1, :])
    gx = (lg[..., :, 1:] - lg[..., :, :-1]).abs() * (v[..., :, 1:] * v[..., :, :-1])
    g = F.pad(gy, (0, 0, 0, 1)) + F.pad(gx, (0, 1))
    w = F.max_pool2d(g, dilate, stride=1, padding=dilate // 2)   # boundary band
    w = w * v
    w = w / (w.sum() / v.sum().clamp(min=1)).clamp(min=1e-6)     # mean 1
    return ((lp - lg).abs() * w).sum() / v.sum().clamp(min=1)


def near_weighted_loss(pred, gt, valid, near=2.0, gain=4.0):
    """Log-depth L1 weighted towards close range (PLAN_ACC A7).

    REPORT §4.43 (A6) split the holdout by region against a 343M reference on
    the same pixels. Our error is 1.13x the reference overall and 1.15x in the
    depth-boundary band -- but 2.38x inside 2 m and 2.66x on independently
    moving pixels. Those two regions are largely the same pixels: 73% of the
    dynamic pixels sit inside 2 m, in both TUM and Bonn. So a depth band is a
    73%-effective dynamic mask that costs one comparison against GT, where an
    actual motion mask costs an optical-flow pass per step.

    That the ratio is measured against another model is what makes it a real
    finding rather than an artefact of AbsRel: dividing by depth inflates the
    near-range error of every model equally, so the 2.38x is left over after
    that cancels.

    Weight is `gain` inside the band and 1 outside, renormalized to mean 1 over
    valid pixels -- so this changes where the loss looks, not how loud it is,
    and the weight can be compared across scenes with different depth ranges.

    ponytail: a band, not a motion mask. The 27% of dynamic pixels beyond 2 m
    are not covered; if a run moves the near-range number and leaves the
    dynamic one behind, the next step is a mask off `warp_residual_loss`'s
    existing flow (its GT warp residual is already a motion signal) rather than
    a second RAFT pass.
    """
    lp, lg = pred.clamp(min=1e-3).log(), gt.clamp(min=1e-3).log()
    v = valid
    w = torch.where(gt < near, gain, 1.0) * v
    w = w / (w.sum() / v.sum().clamp(min=1)).clamp(min=1e-6)     # mean 1
    return ((lp - lg).abs() * w).sum() / v.sum().clamp(min=1)


def bin_ce_loss(logits, centres, gt, valid):
    """Soft cross-entropy tying the depth-bin distribution to the GT bin.

    Measured on v8 (bins=64): the head's distribution has normalized entropy
    0.77 and a 1.02 log-depth sigma, identical at depth boundaries and in flat
    regions, and 99.9% of pixels have no bin holding half the mass. With the
    supervision only on the expectation, a broad distribution whose mean is
    right is a perfectly good optimum, so binning degenerates into scalar
    regression through a softmax bottleneck. AdaBins/BinsFormer avoid this by
    supervising the distribution itself; this is that term.

    logits: (N, bins, h, w) raw head output. centres: (bins,) log-depth, must
    be ascending. gt/valid: (N, 1, H, W) at any resolution >= (h, w).
    Target mass is split linearly between the two centres bracketing log(gt),
    which keeps the loss sub-bin-accurate instead of snapping to one index.
    Centres are detached: the CE shapes the distribution, and the bin
    positions keep being learned by the depth losses through the expectation.
    """
    n, bins, h, w = logits.shape
    c = centres.detach()
    g = F.adaptive_avg_pool2d(gt * valid, (h, w))
    v = F.adaptive_avg_pool2d(valid, (h, w))
    keep = v > 0.99                                   # fully-valid cells only
    if not bool(keep.any()):
        return logits.sum() * 0
    lg = torch.log((g / v.clamp(min=1e-6)).clamp(min=1e-3))
    hi = torch.searchsorted(c, lg.flatten().contiguous()).clamp(1, bins - 1)
    lo = hi - 1
    t = ((lg.flatten() - c[lo]) / (c[hi] - c[lo])).clamp(0, 1)
    logp = F.log_softmax(logits, 1).permute(0, 2, 3, 1).reshape(-1, bins)
    ce = -((1 - t) * logp.gather(1, lo[:, None])[:, 0]
           + t * logp.gather(1, hi[:, None])[:, 0])
    k = keep.flatten()
    return (ce * k).sum() / k.sum().clamp(min=1)


def _norm_field(x, valid, per_sample=False):
    """Median-centre, mean-absolute-deviation scale, over valid pixels.

    per_sample=True does it per frame, which is MiDaS's rule; see `_norm_disp`
    for why the batch-wide alternative quietly silences narrow-range frames."""
    if not per_sample:
        v = valid.bool()
        if not bool(v.any()):
            return x * 0
        t = x[v].median()
        s = (x[v] - t).abs().mean().clamp(min=1e-6)
        return (x - t) / s
    from .sharpness import norm_field

    shp = x.shape
    out = norm_field(x.reshape(-1, 1, *shp[-2:]),
                     valid.reshape(-1, 1, *shp[-2:]))
    return out.reshape(shp)


def _pyramid_grad(p, g, valid, scales=4):
    """Gradient matching across a resolution pyramid, MiDaS's L_reg. Both
    fields must already be normalized. Single-scale matching only sees 1-pixel
    edges; the pyramid also penalizes low-frequency shape error, which is what
    makes high-precision depth models look sharp instead of blurry."""
    # collapse leading dims so avg_pool2d sees (N,1,H,W)
    p, g, v = (x.reshape(-1, 1, *x.shape[-2:]) for x in (p, g, valid))
    total = p.sum() * 0
    for k in range(scales):
        if k:
            p, g = F.avg_pool2d(p * v, 2), F.avg_pool2d(g * v, 2)
            vn = F.avg_pool2d(v, 2)
            p, g = p / vn.clamp(min=1e-6), g / vn.clamp(min=1e-6)
            v = (vn > 0.99).float()   # keep only fully-valid coarse pixels
        if min(p.shape[-2:]) < 4:
            break
        total = total + grad_loss(p, g, v)
    return total / scales


def _norm_disp(depth, valid, per_sample=False):
    """Scale-shift normalized disparity, MiDaS style: median-center and
    mean-absolute-deviation scale, computed over valid pixels only. Puts every
    scene on one comparable footing, which matters here because TartanAir V2
    spans 0.5-129 m in a single frame while Bonn spans 1.5-4 m.

    per_sample=False is the original behaviour and normalizes over the WHOLE
    tensor -- one median and one scale for every frame in the batch. MiDaS
    normalizes per image, and the difference is not cosmetic here: batches are
    drawn from a five-source mixture, so a TartanAir frame's 0.008-2 disparity
    sets the scale and an indoor frame's 0.25-0.67 shrinks to a few percent of
    it. The gradient term that is supposed to supply sharpness then contributes
    almost nothing on exactly the real footage the paper reports."""
    return _norm_field(1.0 / depth.clamp(min=1e-3), valid, per_sample)


def multiscale_grad_loss(pred, gt, valid, scales=4, per_sample=False):
    """Gradient matching on normalized disparity across a resolution pyramid
    (MiDaS's L_reg).

    per_sample normalizes each frame on its own, as MiDaS does -- see
    `_norm_disp`. It is off by default because the reported checkpoint was
    trained without it."""
    return _pyramid_grad(_norm_disp(pred, valid, per_sample),
                         _norm_disp(gt, valid, per_sample), valid, scales)


def spread_loss(pred, gt, valid, min_px=64, eps=1e-6):
    """Penalise dynamic-range compression of the predicted depth field.

    REPORT 4.32: on the dynamic-object source the model produces less than
    half the ground truth's dynamic range (0.47 of it, against 0.92 on the
    static source), regressing toward the mean on exactly the scenes whose
    true range is widest. Nothing in the existing objective penalises that.
    SI-log is scale-invariant and does penalise the mismatch indirectly, but
    under uncertainty shrinking the prediction still lowers it -- the usual
    bias-variance trade -- so compression survives training.

    This measures spread the way the diagnostic did: the standard deviation of
    log depth over valid pixels, per sample rather than per batch, because the
    compression is a property of a scene and averaging over a batch would let a
    wide scene cancel a narrow one. The ratio is compared in log space so the
    term is scale-free and symmetric: over-spreading is penalised as much as
    under-spreading, which matters because a model can otherwise buy this loss
    with noise.

    Samples with almost no valid ground truth are dropped rather than clamped;
    a standard deviation over a handful of pixels is not a range estimate.
    """
    lp = torch.log(pred.clamp(min=eps)).flatten(1)
    lg = torch.log(gt.clamp(min=eps)).flatten(1)
    v = valid.flatten(1).to(lp.dtype)
    n = v.sum(1)
    keep = n >= min_px
    if not bool(keep.any()):
        return pred.sum() * 0.0
    lp, lg, v, n = lp[keep], lg[keep], v[keep], n[keep].unsqueeze(1)

    def std(x):
        m = (x * v).sum(1, keepdim=True) / n
        return ((((x - m) ** 2) * v).sum(1, keepdim=True) / n).clamp(min=eps).sqrt()

    return (torch.log(std(lp) / std(lg)) ** 2).mean()


def boundary_location_loss(pred, gt, valid, decile=0.9, dilate=3):
    """Put the gradient where the GT has one, and nowhere else.

    The measured defect this targets (REPORT 4.44): boundary precision 0.565
    against DPT-Large's 0.740, flat-region TV 37% above it, and the worst
    overshoot of every arm measured. All three are the same failure -- gradient
    energy in the wrong places -- and none of them is what `grad_loss` or
    `edge_weighted_loss` penalize, which are both symmetric in where the error
    sits.

    Two asymmetric terms on per-frame normalized disparity (so it is free of
    scale and shift, like the metric it is written against):

      flat    L1 on the prediction's own gradient outside the GT boundary band.
              Not a difference from the GT's gradient: in a flat region the GT
              IS quiet, and asking for its exact quantized texture back is how
              noise gets rewarded.
      edge    hinge relu(|grad gt| - |grad pred|) inside the band -- reach the
              GT's sharpness, with no reward for exceeding it. A symmetric term
              here would buy the gradient ratio with ringing, which is exactly
              the trade the P2 gate exists to refuse.
    """
    from sokkanaem.sharpness import boundary_band, grad_mag

    shp = (-1, 1) + tuple(pred.shape[-2:])
    p, g, v = (x.reshape(shp) for x in (pred, gt, valid))
    gp, defined = grad_mag(_norm_field(1.0 / p.clamp(min=1e-3), v, True), v)
    gg, _ = grad_mag(_norm_field(1.0 / g.clamp(min=1e-3), v, True), v)
    band = boundary_band(g, v, decile, dilate)
    d = defined.bool()
    flat = d & ~band
    edge = d & band
    n_f, n_e = flat.sum().clamp(min=1), edge.sum().clamp(min=1)
    return ((gp * flat).sum() / n_f
            + ((gg - gp).clamp(min=0) * edge).sum() / n_e)


def rank_loss(pred, gt, valid, pairs=4096, margin=0.05, generator=None):
    """Pairwise depth ordering, on random pixel pairs within each frame.

    Every other term here is a per-pixel regression: it can be satisfied by a
    field that is right on average and has no reliable ordering across an
    occlusion, which is what a "which object is in front" reading of the output
    actually needs. This is the ranking loss from the relative-depth literature
    (DIW/Chen et al.).

    Both fields enter as PER-FRAME normalized disparity, which is what makes
    the term gauge-free: a scale+shift of the prediction's disparity cancels in
    the normalization, so it cannot fight the metric supervision over scale.
    Written on raw depth instead, the same loss changes value under an
    alignment that leaves every ordering intact (measured: 0.318 -> 0.375).

    Whether a pair is "ordered" is decided on the GT's log depth with a
    physical margin (5% depth difference by default) rather than in normalized
    units; pairs below it get an L2 pull toward equality, so "these two pixels
    are at the same depth" is supervised too.
    """
    shp = (-1, 1) + tuple(pred.shape[-2:])
    p, g, v = (x.reshape(shp) for x in (pred, gt, valid))
    dp = _norm_field(1.0 / p.clamp(min=1e-3), v, True)
    dg = _norm_field(1.0 / g.clamp(min=1e-3), v, True)
    lg = g.clamp(min=1e-3).log()
    N = p.shape[0]
    hw = p.shape[-1] * p.shape[-2]
    dp, dg, lg = dp.reshape(N, hw), dg.reshape(N, hw), lg.reshape(N, hw)
    vb = v.reshape(N, hw).bool()
    # positions are drawn over ALL pixels and invalid endpoints are dropped by
    # the mask, rather than drawn from each frame's own valid index list: the
    # per-frame version needed a nonzero() and two randint launches per frame
    # and cost 40 ms per training step against 5 ms for this one. Sampled with
    # replacement -- 4096 pairs out of ~65k^2 candidates, so collisions are
    # negligible next to a sort of the frame.
    a = torch.randint(hw, (N, pairs), device=dp.device, generator=generator)
    b = torch.randint(hw, (N, pairs), device=dp.device, generator=generator)
    take = lambda x, i: x.gather(1, i)
    m = take(vb, a) & take(vb, b)
    ordered = (take(lg, a) - take(lg, b)).abs() > margin
    s_gt = torch.sign(take(dg, a) - take(dg, b))
    d_pr = take(dp, a) - take(dp, b)
    loss = (F.softplus(-s_gt * d_pr) * ordered
            + d_pr.pow(2) * ~ordered) * m
    return loss.sum() / m.sum().clamp(min=1)


def dynamic_mask(frames, flow_px=1.5):
    """Independently-moving pixels, per frame, (B,T,1,H,W) bool.

    Optical flow that disagrees with the frame's DOMINANT motion by more than
    `flow_px`: the median flow absorbs ego-motion, so a panning camera does not
    mark the whole scene dynamic. Identical definition to the `dynamic` region
    in scripts/eval_acc.py, deliberately -- a training weight and the metric it
    is supposed to move should not disagree about which pixels are moving.
    """
    from .metrics import _flow

    B, T = frames.shape[:2]
    out = []
    for b in range(B):
        fl = _flow(frames[b, 1:], frames[b, :-1])        # (T-1,2,H,W)
        med = fl.flatten(2).median(-1).values[..., None, None]
        res = (fl - med).pow(2).sum(1, keepdim=True).sqrt()
        out.append(torch.cat([res[:1], res]) > flow_px)  # frame 0 borrows 1
    return torch.stack(out)


@torch.no_grad()
def _dynamic_weight(frames, valid, gain, flow_px):
    dyn = dynamic_mask(frames, flow_px).float() * valid
    w = (1.0 + gain * dyn) * valid
    return w / (w.sum() / valid.sum().clamp(min=1)).clamp(min=1e-6)


def dynamic_weighted_loss(frames, pred, gt, valid, gain=4.0, flow_px=1.5,
                          min_px=128):
    """Log-depth L1 weighted towards independently-moving pixels.

    The measured bottleneck, not a guess: our dynamic-pixel AbsRel is 2.66x
    DPT-Large's while our TUM *static* pixels are at parity with it (REPORT
    §4.43). Every existing weighted term picks its pixels from GT depth (an
    edge band, a near band), and a moving object in the middle of a room is in
    neither.

    The flow is a second RAFT pass on top of `warp_residual_loss`'s, which is
    the honest cost of an arm that has to be measured before it is optimized.
    # ponytail: share one flow between the two terms if this arm is promoted
    """
    if frames.shape[1] < 2 or min(pred.shape[-2:]) < min_px:
        return pred.sum() * 0                     # RAFT needs >=128px
    w = _dynamic_weight(frames, valid, gain, flow_px)
    lp, lg = pred.clamp(min=1e-3).log(), gt.clamp(min=1e-3).log()
    return ((lp - lg).abs() * w).sum() / valid.sum().clamp(min=1)


def overshoot_loss(pred, gt, valid, win=15, margin=0.05, decile=0.9):
    """Penalise a boundary pixel that leaves the local GT range -- ringing.

    `boundary_location_loss` sharpens with a one-sided hinge so that exceeding
    the GT gradient earns nothing, but measurement says that is not enough: at
    weight 3 it lifted the gradient ratio 0.431 -> 0.578 and pushed overshoot
    0.316 -> 0.361, against a gate of 0.226 (REPORT 4.44). A hinge that does
    not PAY for ringing still permits it. This is the differentiable form of
    the metric itself: how far outside the local GT window's range the
    prediction sits, as a fraction of that range, over the boundary band.
    """
    from .sharpness import boundary_band, norm_disp

    shp = (-1, 1) + tuple(pred.shape[-2:])
    p, g, v = (x.reshape(shp) for x in (pred, gt, valid))
    band = boundary_band(g, v, decile) & v.bool()
    if not bool(band.any()):
        return pred.sum() * 0
    pd, gd = norm_disp(p, v), norm_disp(g, v)
    with torch.no_grad():
        # a FINITE sentinel, not finfo.max: a window that is entirely invalid
        # then yields hi = -sentinel, lo = +sentinel and a huge `out` there,
        # and huge * band(=0) is 0 while inf * 0 is NaN -- which is exactly how
        # this term NaN'd on its first real batch. Normalized disparity is
        # O(10), so 1e4 is out of range by three orders and stays finite.
        big = torch.full_like(gd, 1e4)
        hi = F.max_pool2d(torch.where(v.bool(), gd, -big), win, 1, win // 2)
        lo = -F.max_pool2d(torch.where(v.bool(), -gd, -big), win, 1, win // 2)
        rng = (hi - lo).clamp(min=1e-6)
    out = ((pd - hi).clamp(min=0) + (lo - pd).clamp(min=0)) / rng
    return ((out - margin).clamp(min=0) * band).sum() / band.sum().clamp(min=1)
