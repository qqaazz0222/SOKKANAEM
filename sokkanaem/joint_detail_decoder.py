"""Trainable DPT decoder with an optional RGB detail-feature path.

Unlike earlier readout probes, every DPT decoder layer is trainable. The SSM
encoder can remain frozen through cached *pre-decoder* multiblock tokens.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from sokkanaem.detail_feature_path import DetailFeaturePath


class JointDetailDecoder(nn.Module):
    def __init__(self, decoder, detail=True):
        super().__init__()
        if decoder.full_res or not decoder.bins:
            raise ValueError("This study requires the original binned, half-resolution DPT")
        self.decoder = copy.deepcopy(decoder).requires_grad_(True)
        self.detail = DetailFeaturePath() if detail else None

    def forward(self, tokens, rgb):
        d = self.decoder
        x = sum(proj(t) for proj, t in zip(d.proj, tokens))
        skips, h = [], rgb
        for block in d.stem:
            h = block(h); skips.append(h)
        for i, (skip, red, mix) in enumerate(zip(reversed(skips), d.reduce, d.mix)):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            a, b = red(x), skip
            if d.norm_x is not None:
                a, b = d.norm_x[i](a), d.norm_s[i](b)
            x = mix(a + b)
        if self.detail is not None:
            x = self.detail(rgb, x)
        probs = (d.head(x)/d.bin_temp).softmax(1)
        logd = (probs*d.bin_centres()[None,:,None,None]).sum(1,keepdim=True)
        return F.interpolate(logd,scale_factor=2,mode="bilinear",align_corners=False).exp()


def reliable_edge_loss(pred, gt, valid):
    """Signed log-depth gradient on valid GT discontinuities, not RGB texture.

    Only used as a separate, documented training objective in this new study.
    Threshold 0.02 log-depth is fixed before evaluation; sensor edges remain
    imperfect supervision despite eroding the invalid mask.
    """
    keep = valid.bool() & torch.isfinite(gt) & (gt > 0) & (gt < 150)
    keep = F.avg_pool2d(keep.float(),3,1,1) > .999
    p = pred.clamp_min(1e-6).log()
    g = torch.nan_to_num(gt,nan=0.,posinf=0.,neginf=0.).clamp_min(1e-6).log()
    loss = pred.sum()*0
    for axis in (-1,-2):
        a,b=[slice(None)]*4,[slice(None)]*4
        a[axis],b[axis]=slice(1,None),slice(None,-1)
        dp,dg=p.diff(dim=axis),g.diff(dim=axis)
        mask=keep[tuple(a)] & keep[tuple(b)] & (dg.abs()>.02)
        loss=loss+((dp-dg).abs()*mask).sum()/mask.sum().clamp_min(1)
    return loss
