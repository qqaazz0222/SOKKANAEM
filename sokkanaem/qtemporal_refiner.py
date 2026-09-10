"""Two-frame error-gated depth residual; no inference GT or dense Q0 oracle.

The learned gate predicts where coarse outputs need repair, not certified flow,
occlusion or true boundaries. Both temporal/control paths have identical size.
"""
import torch
from torch import nn
import torch.nn.functional as F


class TemporalErrorRefiner(nn.Module):
    def __init__(self, temporal=True, bound=.1):
        super().__init__()
        self.temporal = temporal; self.bound = bound
        self.shallow = nn.Sequential(nn.Conv2d(11, 16, 3, padding=1), nn.GELU())
        self.context = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
                                     nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 2, 1))
        nn.init.zeros_(self.fuse[-1].weight); nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, current, previous, coarse, previous_depth):
        size = coarse.shape[-2:]
        current = F.interpolate(current.float(), size, mode="bilinear", align_corners=False)
        previous = F.interpolate(previous.float(), size, mode="bilinear", align_corners=False)
        previous_depth = F.interpolate(previous_depth.float(), size, mode="bilinear", align_corners=False)
        if not self.temporal:
            previous = current; previous_depth = coarse
        x = torch.cat((current, previous, (current-previous).abs(), coarse.clamp_min(1e-4).log(),
                       previous_depth.clamp_min(1e-4).log()), 1)
        local = self.shallow(x)
        context = F.interpolate(self.context(local), size, mode="bilinear", align_corners=False)
        raw, gate_logit = self.fuse(torch.cat((local, context), 1)).chunk(2, 1)
        delta = self.bound*raw.tanh()*gate_logit.sigmoid()
        return coarse*delta.exp(), gate_logit


@torch.no_grad()
def repair_target(coarse, teacher):
    p, t = coarse.clamp_min(1e-4).log(), teacher.clamp_min(1e-4).log()
    dx = F.pad((torch.diff(p, dim=-1)-torch.diff(t, dim=-1)).abs(), (0, 1))
    dy = F.pad((torch.diff(p, dim=-2)-torch.diff(t, dim=-2)).abs(), (0, 0, 0, 1))
    return (((p-t).abs() > .015) | ((dx+dy) > .02)).float()


class TemporalRefinedStream(nn.Module):
    def __init__(self, stream, refiner, d_min, d_max):
        super().__init__(); self.stream = stream; self.refiner = refiner
        self.d_min = d_min; self.d_max = d_max

    @torch.no_grad()
    def step(self, frame, state=None):
        pred, base, info = self.stream.step(frame, None if state is None else state["base"])
        if not info["full_refresh"]:
            if state is None: raise ValueError("Cannot update without a previous frame")
            pred, _ = self.refiner(frame, state["previous_rgb"], pred, state["previous_depth"])
            pred = pred.clamp(min=self.d_min, max=self.d_max)
        new_state = {"base": base, "previous_rgb": frame.clone(), "previous_depth": pred.clone()}
        return pred, new_state, info
