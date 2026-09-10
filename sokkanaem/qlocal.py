"""Output-local bounded refinement with explicit predicted/RGB edge exclusion."""
import torch
import torch.nn.functional as F
from sokkanaem.qrgb_refiner import QRGBRefiner


@torch.no_grad()
def editable_mask(rgb, depth):
    rgb = F.interpolate(rgb.float(), depth.shape[-2:], mode="bilinear", align_corners=False)
    log = depth.clamp_min(1e-4).log()
    dx = F.pad(torch.diff(log, dim=-1).abs(), (0, 1))
    dy = F.pad(torch.diff(log, dim=-2).abs(), (0, 0, 0, 1))
    rx = F.pad(torch.diff(rgb, dim=-1).abs().amax(1, keepdim=True), (0, 1))
    ry = F.pad(torch.diff(rgb, dim=-2).abs().amax(1, keepdim=True), (0, 0, 0, 1))
    boundary = ((dx+dy) > .02) | ((rx+ry) > .12)
    protected = F.max_pool2d(boundary.float(), 5, 1, 2) > 0
    return ~protected


class LocalRefiner(QRGBRefiner):
    def __init__(self, protected=True):
        super().__init__(bound=.03)
        self.protected = protected

    def forward(self, rgb, depth):
        refined = super().forward(rgb, depth)
        if self.protected:
            return torch.where(editable_mask(rgb, depth), refined, depth)
        return refined
