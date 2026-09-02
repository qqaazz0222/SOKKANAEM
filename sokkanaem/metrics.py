"""Motion-compensated temporal metrics (IDEA.md §4.2).

`t-delta` (raw |D_t - D_{t-1}|) has a trivial global optimum: a constant
output scores exactly 0 — the v1 collapse did precisely that (REPORT.md
§4.6). Flow warping alone does NOT fix this: a spatially constant field is
invariant under any warp, so plain OPW also scores 0 for a constant. Hence
two metrics off one shared warp:

    opw  flow-warped self-consistency, the usual video-depth number
         (NVDS/VDA style). Comparable to prior work; still degenerate for
         constant predictions, so never read it without accuracy alongside.
    tce  the same warp residual measured *against GT's own* warp residual.
         A constant prediction now scores GT's full inter-frame geometry
         change instead of 0, and a perfect prediction scores 0. This is
         the non-degenerate temporal number.

Both are relative (divided by GT depth), so they need the prediction
median-scaled to GT — which is how every model here is already scored.
TAE (pose-warped) is not implemented: it needs per-dataset extrinsics +
intrinsics plumbing that these three adapters don't carry.
"""
import torch
import torch.nn.functional as F

_raft = None


def _flow(img1, img2, chunk=64):
    """RAFT flow img1 -> img2, both (B,3,H,W) in [0,1]. H,W >= 128 and /8
    (RAFT's correlation pyramid needs a >=16px feature map).

    Chunked over the batch: a 512-frame clip is 511 pairs, and RAFT's
    correlation volume for that many at once does not fit in 24 GB. The result
    is identical to one call -- pairs are independent."""
    global _raft
    assert min(img1.shape[-2:]) >= 128, f"RAFT needs >=128px, got {img1.shape[-2:]}"
    if _raft is None:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        _raft = raft_small(weights=Raft_Small_Weights.DEFAULT)
        _raft = _raft.to(img1.device).eval()
    if img1.shape[0] <= chunk:
        return _raft(img1 * 2 - 1, img2 * 2 - 1)[-1]  # last refinement iteration
    return torch.cat([_raft(img1[i:i + chunk] * 2 - 1, img2[i:i + chunk] * 2 - 1)[-1]
                      for i in range(0, img1.shape[0], chunk)])


@torch.no_grad()
def warp_grids(frames):
    """frames (T,3,H,W) in [0,1] -> (grid, inb), the backward warp for each
    consecutive pair. grid (T-1,H,W,2): where each frame-(t+1) pixel came
    from in frame t. inb (T-1,1,H,W): source landed inside the image.
    Computed once from RGB and reused for prediction and GT alike, so the
    warp is identical across every model being compared."""
    T, _, H, W = frames.shape
    flow = _flow(frames[1:], frames[:-1])  # t+1 -> t
    yy, xx = torch.meshgrid(
        torch.arange(H, device=frames.device, dtype=flow.dtype),
        torch.arange(W, device=frames.device, dtype=flow.dtype), indexing="ij")
    sx, sy = xx + flow[:, 0], yy + flow[:, 1]
    inb = ((sx >= 0) & (sx <= W - 1) & (sy >= 0) & (sy <= H - 1))
    grid = torch.stack([sx / (W - 1) * 2 - 1, sy / (H - 1) * 2 - 1], -1)
    return grid, inb.unsqueeze(1).float()


def warp(x, grid):
    """Sample x (T-1,1,H,W) at grid — nearest, so warping a validity mask
    stays 0/1 and warped depth never blends across an occlusion edge."""
    return F.grid_sample(x, grid, mode="nearest", padding_mode="border",
                         align_corners=True)


@torch.no_grad()
def temporal_metrics(frames, pred, gt, valid=None, pooled=False):
    """frames (T,3,H,W) [0,1]; pred, gt (T,1,H,W) depth in the SAME scale
    (median-scale pred first); valid (T,1,H,W) 0/1 or None.
    Returns {"opw": float, "tce": float} — see module docstring. pooled=True
    additionally returns the unnormalized sums and the valid-pixel count, for
    dataset-level (pixel-weighted) aggregation.

    ponytail: occlusions are only handled by the in-bounds test and the GT
    validity warp, not a forward-backward flow consistency check. Both
    metrics see the same mask for every model, so the comparison is fair;
    add the fb-check if absolute values ever need to match a paper's.
    """
    grid, inb = warp_grids(frames)
    m = inb
    if valid is not None:
        m = m * valid[1:] * warp(valid[:-1], grid)  # source AND target valid
    denom = gt[1:].clamp(min=1e-6)
    dp = warp(pred[:-1], grid) - pred[1:]
    dg = warp(gt[:-1], grid) - gt[1:]
    n = m.sum().clamp(min=1)
    opw_sum = ((dp.abs() / denom) * m).sum().item()
    tce_sum = (((dp - dg).abs() / denom) * m).sum().item()
    out = {"opw": opw_sum / n.item(), "tce": tce_sum / n.item()}
    if pooled:
        out.update(opw_sum=opw_sum, tce_sum=tce_sum, warp_px=n.item())
    return out


def scale_stats(pred, gt, valid):
    """Per-frame alignment factors s_t = median(gt_t)/median(pred_t) and what
    they do over a clip (r2 Major 1). pred may already be clip-aligned: every
    statistic here is a ratio or a log-difference, so a constant clip-level
    scale cancels and what is left is the model's own scale stability.

    A long-clip penalty under one-scale-per-clip alignment decomposes into
    depth-shape drift and drift of s_t itself. The second is a property of the
    model, not of the protocol, which is why it is reported next to the error:

        drift    coefficient of variation of s_t  (kept for continuity)
        logstd   std of log s_t -- the scale-symmetric version of drift
        step     mean |log s_t - log s_{t-1}| -- frame-to-frame scale jitter,
                 the part a per-frame-aligned metric removes by construction
    """
    fs = [(gt[t][vt].median() / pred[t][vt].median().clamp(min=1e-6)).item()
          for t in range(gt.shape[0]) if (vt := valid[t].bool()).any()]
    if len(fs) < 2:
        return {"scale_drift": 0.0, "scale_logstd": 0.0, "scale_step": 0.0}
    fs = torch.tensor(fs).clamp(min=1e-12)
    lf = fs.log()
    return {"scale_drift": (fs.std() / fs.mean().clamp(min=1e-6)).item(),
            "scale_logstd": lf.std().item(),
            "scale_step": (lf[1:] - lf[:-1]).abs().mean().item()}


@torch.no_grad()
def clip_scores(frames, pred, gt, valid, temporal=True):
    """Every per-clip number for one clip, computed in ONE place so eval.py
    and the baseline scripts cannot drift apart — they already had (t-delta
    was measured on raw output in eval.py and on median-scaled output in the
    baselines, REPORT.md §4.10).

    frames (T,3,H,W) in [0,1]; pred, gt (T,1,H,W) with pred ALREADY aligned
    to gt's scale; valid (T,1,H,W) 0/1.

    temporal=False drops OPW/TCE (and the RAFT pass that costs more than every
    other metric here combined). PLAN_ACC's G1 is an accuracy gate and does not
    read them; G2/G4 do, so the default stays on.

    Returns the per-clip means AND, under "_pooled", the raw sums+counts so
    callers can report the pixel-pooled dataset-level metric that the depth
    literature uses. The two differ a lot when a model fails on a few clips:
    DA v2's per-clip AbsRel mean is 0.53 with std 2.03 — the mean is carried
    by a handful of blown-up clips, so quoting only per-clip means would
    misrepresent it in either direction.
    """
    v = valid.bool()
    if not bool(v.any()):
        # Real depth sensors drop whole frames (Kinect dropouts, and a center
        # crop can land entirely in an invalid region). Such a clip has no
        # median to scale by, so every metric is nan — and t-delta/OPW/TCE are
        # measured over ALL pixels, so one poisoned clip nans the pooled sum
        # for the entire dataset. Skip it and let the caller count it.
        return None
    p, g = pred[v], gt[v]
    gc = g.clamp(min=1e-6)
    rel = (p - g).abs() / gc
    sq = (p - g) ** 2
    r = torch.maximum(p / gc, g / p.clamp(min=1e-6))
    d1 = (r < 1.25).float()
    td = (pred[1:] - pred[:-1]).abs()
    out = {"absrel": rel.mean().item(), "rmse": sq.mean().sqrt().item(),
           "delta1": d1.mean().item(), "temporal_delta": td.mean().item()}
    tm = (temporal_metrics(frames, pred, gt, valid, pooled=True) if temporal
          else {"opw": 0.0, "tce": 0.0, "opw_sum": 0.0, "tce_sum": 0.0,
                "warp_px": 0.0})
    out.update({k: tm[k] for k in ("opw", "tce")})
    out.update(scale_stats(pred, gt, valid))
    out["_pooled"] = {
        "rel_sum": rel.sum().item(), "sq_sum": sq.sum().item(),
        "d1_sum": d1.sum().item(), "px": float(v.sum().item()),
        "td_sum": td.sum().item(), "td_px": float(td.numel()),
        "opw_sum": tm["opw_sum"], "tce_sum": tm["tce_sum"],
        "warp_px": tm["warp_px"],
    }
    return out


def robust(values):
    """Mean is not enough when a handful of clips blow up: PLAN_ACC A3.

    DA V2 Small reads 0.2256 mean against 0.0967 clip median on the same clips
    -- one statistic says "twice as bad as us", the other "slightly better".
    Both are true of the same numbers, so the table prints both and the trimmed
    mean and tail between them, and nothing is selected after the fact.
    """
    import statistics
    x = sorted(values)
    n = len(x)
    if n == 0:
        return {k: float("nan") for k in
                ("mean", "median", "trim10", "p90", "p95", "max")} | {"n": 0}
    k = int(n * 0.05)               # 10% trimmed = 5% off each end
    core = x[k:n - k] or x

    def pct(q):
        i = min(n - 1, max(0, int(round(q * (n - 1)))))
        return x[i]

    return {"mean": statistics.fmean(x), "median": statistics.median(x),
            "trim10": statistics.fmean(core), "p90": pct(0.90),
            "p95": pct(0.95), "max": x[-1], "n": n}


def boot_ci(per_source, reps=10000, seed=0, lo=2.5, hi=97.5, stat="mean"):
    """95% percentile CI of the dataset-BALANCED statistic, clips resampled
    within each source (PLAN_ACC A3, G1's "CI must not overlap" clause).

    per_source: {source: [per-clip values]}. Stratified because the reported
    number is a mean over sources, not over clips -- resampling the pooled list
    would let Bonn's 399 clips outvote TUM's 89 and understate the interval.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    f = np.median if stat == "median" else np.mean
    draws = []
    for v in per_source.values():
        a = np.asarray(v, dtype=float)
        idx = rng.integers(0, len(a), size=(reps, len(a)))
        draws.append(f(a[idx], axis=1))
    b = np.mean(draws, axis=0)
    return float(np.percentile(b, lo)), float(np.percentile(b, hi))


def pooled(sums):
    """Aggregate the "_pooled" dicts of many clips into dataset-level numbers
    (pixel-weighted), the convention in the depth literature."""
    t = {}
    for s in sums:
        for k, v in s.items():
            t[k] = t.get(k, 0.0) + v
    px, wpx = max(t["px"], 1.0), max(t["warp_px"], 1.0)
    return {"absrel": t["rel_sum"] / px,
            "rmse": (t["sq_sum"] / px) ** 0.5,
            "delta1": t["d1_sum"] / px,
            "temporal_delta": t["td_sum"] / max(t["td_px"], 1.0),
            "opw": t["opw_sum"] / wpx,
            "tce": t["tce_sum"] / wpx}


def report(label, acc):
    """Summary for the baseline scripts, same columns as scripts/eval.py.
    Reports the pixel-pooled dataset-level metric first (the convention, and
    robust to a few catastrophic clips) with the per-clip mean±std beside it,
    and dumps the per-clip values so a different statistic never needs a
    re-run of the model."""
    import json
    import re
    import statistics
    from pathlib import Path

    n = len(acc["absrel"])
    pl = pooled(acc["_pooled"])
    sd = statistics.stdev(acc["absrel"]) if n > 1 else 0.0
    print(f"{label} on {n} holdout clips (same split/protocol as SOKKANAEM):")
    print(f"  pooled : AbsRel={pl['absrel']:.4f}  RMSE={pl['rmse']:.4f}  "
          f"delta1={pl['delta1']:.4f}  t-delta={pl['temporal_delta']:.4f}  "
          f"OPW={pl['opw']:.4f}  TCE={pl['tce']:.4f}")
    print(f"  clipavg: AbsRel={sum(acc['absrel'])/n:.4f} (std {sd:.4f})  "
          f"delta1={sum(acc['delta1'])/n:.4f}")
    out = Path("work_dirs/baselines")
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    path = out / f"{slug}.json"
    path.write_text(json.dumps(
        {k: v for k, v in acc.items() if k != "_pooled"}))
    print(f"  per-clip values -> {path}")


def _selfcheck():
    """The one property the r2 scale decomposition rests on: these statistics
    measure the model's scale wobble, NOT the clip-level alignment, so they
    must not move when the whole prediction is rescaled."""
    g = torch.ones(6, 1, 4, 4)
    v = torch.ones(6, 1, 4, 4)
    p = torch.ones(6, 1, 4, 4)
    p[:, 0, 0, 0] = 1.0  # constant scale -> zero drift, zero step
    a = scale_stats(p, g, v)
    assert max(abs(x) for x in a.values()) < 1e-6, a
    p = torch.stack([torch.full((1, 4, 4), f) for f in
                     (1.0, 2.0, 1.0, 2.0, 1.0, 2.0)])
    b, c = scale_stats(p, g, v), scale_stats(p * 7.3, g, v)
    step = abs(torch.tensor(2.0).log().item())
    assert abs(b["scale_step"] - step) < 1e-5, b
    assert all(abs(b[k] - c[k]) < 1e-5 for k in ("scale_logstd", "scale_step")), (b, c)
    assert abs(b["scale_drift"] - c["scale_drift"]) < 1e-5, (b, c)
    print("scale_stats ok:", {k: round(v, 4) for k, v in b.items()})


if __name__ == "__main__":
    _selfcheck()
