"""Small current-RGB log-depth residual, for skipped frames only."""
import torch
from torch import nn
import torch.nn.functional as F


class QRGBRefiner(nn.Module):
    def __init__(self, bound=.06):
        super().__init__()
        self.bound = bound
        self.net = nn.Sequential(nn.Conv2d(6, 16, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(16, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, rgb, depth):
        rgb = F.interpolate(rgb.float(), depth.shape[-2:], mode="bilinear", align_corners=False)
        log = depth.clamp_min(1e-4).log()
        dx = F.pad(torch.diff(log, dim=-1), (0, 1))
        dy = F.pad(torch.diff(log, dim=-2), (0, 0, 0, 1))
        delta = self.bound*self.net(torch.cat((rgb, log, dx, dy), 1)).tanh()
        return depth*delta.exp()


class RefinedStream(nn.Module):
    def __init__(self, stream, refiner):
        super().__init__(); self.stream = stream; self.refiner = refiner

    @torch.no_grad()
    def step(self, frame, state=None):
        pred, state, info = self.stream.step(frame, state)
        if not info["full_refresh"]:
            pred = self.refiner(frame, pred)
        return pred, state, info
