"""Experimental anchored log-depth mixture readout on frozen DPT features.

The original bin head supplies a frozen initialization anchor, not a teacher
label at inference. Mean and mode readouts use exactly the same learned mixture.
This is a staged readout experiment, not a completely replaced depth backbone.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class AnchoredMixtureReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(16, 5, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[1] = math.log(math.expm1(.02))
            self.net[-1].bias[3:] = math.log(math.expm1(.1))

    def forward(self, features, coarse):
        raw = F.interpolate(self.net(features), size=coarse.shape[-2:],
                            mode="bilinear", align_corners=False)
        center = coarse.clamp_min(1e-6).log() + .5 * raw[:, :1].tanh()
        sep = F.softplus(raw[:, 1:2]).clamp_max(1.)
        means = torch.cat((center-sep, center+sep), 1)
        scales = F.softplus(raw[:, 3:5]).clamp(.005, 1.)
        logits = torch.cat((raw[:, 2:3], torch.zeros_like(raw[:, 2:3])), 1)
        logpi = logits.log_softmax(1)
        mean = (logpi.exp()*means).sum(1, keepdim=True).exp()
        # Evaluate full mixture density at both component centers. Choosing by
        # weight alone ignores component scale and is NOT this decision rule.
        density = logpi[:, None] - (means[:, :, None]-means[:, None, :]).abs()/scales[:, None] - (2*scales[:, None]).log()
        choice = density.logsumexp(2).argmax(1, keepdim=True)
        mode = means.gather(1, choice).exp()
        return {"mean": mean, "mode": mode, "means": means, "scales": scales,
                "logpi": logpi, "choice": choice}


def supervised_loss(pred, gt, valid, mixture=None):
    valid = valid.bool() & torch.isfinite(gt) & (gt > 0) & (gt < 150)
    valid = F.avg_pool2d(valid.float(), 3, 1, 1) > .999
    target = torch.nan_to_num(gt, nan=0., posinf=0., neginf=0.).clamp_min(1e-6).log()
    lp = pred.clamp_min(1e-6).log()
    def avg(x, v): return (x*v).sum()/v.sum().clamp_min(1)
    metric = avg((lp-target).abs(), valid)
    gradient = pred.sum()*0
    for axis in (-1, -2):
        a,b = [slice(None)]*4,[slice(None)]*4
        a[axis],b[axis] = slice(1,None),slice(None,-1)
        v = valid[tuple(a)] & valid[tuple(b)]
        gradient = gradient + avg((lp.diff(dim=axis)-target.diff(dim=axis)).abs(),v)
    loss = metric + .5*gradient
    nll = loss*0
    if mixture is not None:
        nllmap = -(mixture["logpi"] - (target-mixture["means"]).abs()/mixture["scales"]
                   - (2*mixture["scales"]).log()).logsumexp(1,keepdim=True)
        nll = avg(nllmap,valid)
        loss = loss + .05*nll
    return loss, {"metric": float(metric.detach()), "gradient":float(gradient.detach()),
                   "nll":float(nll.detach())}
