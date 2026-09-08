"""Experimental, identity-initialized refinement; not part of frozen v11.

Requires the 16-channel tensor entering DPTDecoder.head. The residual changes
depth by at most exp(+-max_log_delta); the learned gate uses no ground truth.
"""
import torch
from torch import nn
import torch.nn.functional as F


class BoundedDepthRefiner(nn.Module):
    def __init__(self, feature_channels=16, width=16, max_log_delta=0.15):
        super().__init__()
        if not 0 < max_log_delta <= 0.5:
            raise ValueError("max_log_delta must be in (0, 0.5]")
        self.max_log_delta = float(max_log_delta)
        self.semantic = nn.Conv2d(feature_channels, 8, 1)
        self.body = nn.Sequential(
            nn.Conv2d(14, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width), nn.GELU(),
            nn.Conv2d(width, width, 1), nn.GELU())
        self.head = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, rgb, coarse, features):
        logd = coarse.clamp_min(1e-6).log()
        dx = F.pad(logd[..., 1:] - logd[..., :-1], (0, 1))
        dy = F.pad(logd[..., 1:, :] - logd[..., :-1, :], (0, 0, 0, 1))
        semantic = F.interpolate(self.semantic(features), size=coarse.shape[-2:],
                                 mode="bilinear", align_corners=False)
        out = self.head(self.body(torch.cat((rgb, logd, dx, dy, semantic), 1)))
        delta = self.max_log_delta * out[:, :1].tanh() * out[:, 1:].sigmoid()
        return coarse * delta.exp(), delta


def refinement_loss(pred, coarse, gt, valid, delta, edge_weight=0.5):
    """Metric + signed log-gradient matching, excluding 1px invalid borders.

    Preservation is strongest away from GT edges, only during training.
    This is validity-aware supervision, NOT a teacher-confidence estimator.
    """
    keep = valid.bool() & torch.isfinite(gt) & (gt > 0) & (gt < 150)
    keep = F.avg_pool2d(keep.float(), 3, 1, 1) > 0.999
    safe_gt = torch.nan_to_num(gt, nan=0., posinf=0., neginf=0.)
    lp, lg = pred.clamp_min(1e-6).log(), safe_gt.clamp_min(1e-6).log()
    def avg(x, mask):
        return (x * mask).sum() / mask.sum().clamp_min(1)
    metric = avg((lp - lg).abs(), keep)
    grad = pred.sum() * 0
    flat = keep.clone()
    for axis in (-1, -2):
        dp, dg = lp.diff(dim=axis), lg.diff(dim=axis)
        a, b = [slice(None)] * 4, [slice(None)] * 4
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        mask = keep[tuple(a)] & keep[tuple(b)]
        grad = grad + avg((dp - dg).abs(), mask)
        flat[tuple(a)] &= dg.abs() < 0.02
        flat[tuple(b)] &= dg.abs() < 0.02
    preserve = avg(delta.abs(), flat)
    total = metric + edge_weight * grad + 0.1 * preserve + 0.01 * delta.square().mean()
    return total, {"metric": float(metric.detach()), "gradient": float(grad.detach()),
                   "preserve": float(preserve.detach())}
