"""Local previous-depth candidate transport; no inference flow/GT oracle."""
import torch
import torch.nn.functional as F
from torch import nn
from sokkanaem.qtemporal_refiner import TemporalErrorRefiner


OFFSETS = (-4, -2, -1, 0, 1, 2, 4)


def candidate_bank(coarse, previous_depth):
    previous_depth = F.interpolate(previous_depth, coarse.shape[-2:], mode="bilinear", align_corners=False)
    h, w = coarse.shape[-2:]
    padded = F.pad(previous_depth, (4, 4, 4, 4), mode="replicate")
    return torch.cat([coarse]+[padded[..., 4+dy:4+dy+h, 4+dx:4+dx+w]
                              for dy in OFFSETS for dx in OFFSETS], 1)


@torch.no_grad()
def candidate_target(bank, teacher):
    errors = (bank.clamp_min(1e-4).log()-teacher.clamp_min(1e-4).log()).abs()
    minimum, index = errors.min(1)
    # Near ties should not induce arbitrary motion in otherwise stable areas.
    return torch.where(errors[:, 0]-minimum > .005, index, torch.zeros_like(index))


class TransportRefiner(TemporalErrorRefiner):
    def __init__(self, selection="hard"):
        super().__init__(temporal=True)
        self.selection = selection
        self.fuse[-1] = nn.Conv2d(16, 51, 1)
        nn.init.zeros_(self.fuse[-1].weight); nn.init.zeros_(self.fuse[-1].bias)
        with torch.no_grad(): self.fuse[-1].bias[0] = 4.

    def forward(self, current, previous, coarse, previous_depth):
        size = coarse.shape[-2:]
        current = F.interpolate(current.float(), size, mode="bilinear", align_corners=False)
        previous = F.interpolate(previous.float(), size, mode="bilinear", align_corners=False)
        previous_depth = F.interpolate(previous_depth.float(), size, mode="bilinear", align_corners=False)
        x = torch.cat((current, previous, (current-previous).abs(), coarse.clamp_min(1e-4).log(),
                       previous_depth.clamp_min(1e-4).log()), 1)
        local = self.shallow(x)
        context = F.interpolate(self.context(local), size, mode="bilinear", align_corners=False)
        out = self.fuse(torch.cat((local, context), 1)); logits, residual = out[:, :50], out[:, 50:]
        bank = candidate_bank(coarse, previous_depth)
        if self.selection == "disabled":
            selected = coarse
        elif self.selection == "soft":
            selected = (logits.softmax(1)*bank.clamp_min(1e-4).log()).sum(1, keepdim=True).exp()
        elif self.selection == "hard":
            selected = bank.gather(1, logits.argmax(1, keepdim=True))
            if self.training:
                # Straight-through soft surrogate: biased estimator, not exact
                # differentiation of argmax. Forward still selects one depth.
                soft = (logits.softmax(1)*bank.clamp_min(1e-4).log()).sum(1, keepdim=True).exp()
                selected = selected+(soft-soft.detach())
        else:
            raise ValueError("Unknown selection policy")
        return selected*(.03*residual.tanh()).exp(), logits
