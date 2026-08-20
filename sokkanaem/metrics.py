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


@torch.no_grad()
def clip_scores(frames, pred, gt, valid):
    """Every per-clip number for one clip, computed in ONE place so eval.py
    and the baseline scripts cannot drift apart — they already had (t-delta
    was measured on raw output in eval.py and on median-scaled output in the
    baselines, REPORT.md §4.10).

    frames (T,3,H,W) in [0,1]; pred, gt (T,1,H,W) with pred ALREADY aligned
    to gt's scale; valid (T,1,H,W) 0/1.

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
    tm = temporal_metrics(frames, pred, gt, valid, pooled=True)
    out.update({k: tm[k] for k in ("opw", "tce")})
    out["_pooled"] = {
        "rel_sum": rel.sum().item(), "sq_sum": sq.sum().item(),
        "d1_sum": d1.sum().item(), "px": float(v.sum().item()),
        "td_sum": td.sum().item(), "td_px": float(td.numel()),
        "opw_sum": tm["opw_sum"], "tce_sum": tm["tce_sum"],
        "warp_px": tm["warp_px"],
    }
    return out


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
