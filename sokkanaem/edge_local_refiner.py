"""Second screening hypothesis: local corrections rather than depth drift."""
import torch
import torch.nn.functional as F
from sokkanaem.bounded_refiner import BoundedDepthRefiner, refinement_loss


class EdgeLocalRefiner(BoundedDepthRefiner):
    def forward(self, rgb, coarse, features):
        _, raw = super().forward(rgb, coarse, features)
        # Difference from a local mean rejects low-frequency metric corrections.
        smooth = F.avg_pool2d(F.pad(raw, (2, 2, 2, 2), mode="replicate"), 5, 1)
        ld = coarse.clamp_min(1e-6).log()
        dx = F.pad(ld.diff(dim=-1).abs(), (0, 1))
        dy = F.pad(ld.diff(dim=-2).abs(), (0, 0, 0, 1))
        # Soft geometric prior: RGB texture alone cannot open this gate.
        gate = (F.max_pool2d(dx + dy, 5, 1, 2) / .04).clamp(0, 1)
        delta = .5 * (raw - smooth) * gate
        return coarse * delta.exp(), delta


def edge_local_loss(pred, coarse, gt, valid, delta, edge_weight=1.):
    gt = torch.nan_to_num(gt, nan=0., posinf=0., neginf=0.)
    base, parts = refinement_loss(pred, coarse, gt, valid, delta, edge_weight=0.)
    keep = (gt > 0) & (gt < 150) & valid.bool()
    keep = F.avg_pool2d(keep.float(), 3, 1, 1) > .999
    p, c, g = [x.clamp_min(1e-6).log() for x in (pred, coarse, gt)]
    edge, flat = pred.sum()*0, pred.sum()*0
    for axis in (-1, -2):
        dp, dc, dg = [x.diff(dim=axis) for x in (p, c, g)]
        a, b = [slice(None)]*4, [slice(None)]*4
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        mask = keep[tuple(a)] & keep[tuple(b)]
        boundary = mask & (dg.abs() > .02)
        smooth = mask & (dg.abs() < .005)
        edge = edge + ((dp-dg).abs()*boundary).sum()/boundary.sum().clamp_min(1)
        flat = flat + ((dp.abs()-dc.abs()).relu()*smooth).sum()/smooth.sum().clamp_min(1)
    parts.update(edge=float(edge.detach()), flat_excess=float(flat.detach()))
    return base + edge_weight*edge + 2.*flat, parts
