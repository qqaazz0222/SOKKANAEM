"""Matched aligned-feature residual heads, no unused global calibration head.

K2 refresh resets h: an SSM here is a one-step gated parameterization, not
evidence of recurrent memory. MLP control differs by three core parameters.
"""
import torch
import torch.nn.functional as F
from torch import nn
from sokkanaem.qdelta_ssm import QDeltaSSM


class PatchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 77), nn.GELU(), nn.Linear(77, 32))

    def step(self, u, h=None):
        return self.net(u), torch.zeros_like(h)


def zero_state_formula(core, u):
    """Nonrecurrent algebra of SSM with h0=0; float reductions can differ."""
    x, z, dt, bp, cp = core._params(u[:, None], None)
    h = (dt*x).unsqueeze(-1)*bp.unsqueeze(2)
    y = (h*cp.unsqueeze(2)).sum(-1)
    return core._finish(y, x, z)[:, 0], h[:, 0]


class AlignedAdapter(QDeltaSSM):
    def __init__(self, mode="ssm"):
        if mode not in ("ssm", "mlp"):
            raise ValueError("Unsupported core")
        super().__init__()
        del self.global_readout
        if mode == "mlp": self.ssm = PatchMLP()
        self.mode = mode

    def update(self, frame, mask, state):
        b, n, c = state["features"][-1].shape
        if b != 1 or mask.shape != (1, n-1):
            raise ValueError("One stream with patch mask required")
        idx = (mask.flatten() > .5).nonzero().flatten()
        if not len(idx): return state, {"updated_patches": 0}
        rgb = F.interpolate(frame, (state["grid"][0]*14, state["grid"][1]*14), mode="bilinear", align_corners=False)
        local = self.rgb(rgb).flatten(2).transpose(1, 2)
        context = self.context(state["features"][-1][:, 1:])
        u = self.fuse(torch.cat((local, context), -1)).reshape(-1, self.width)
        y, h = self.ssm.step(u[idx], h=state["h"][idx])
        hidden = state["h"].clone().index_copy(0, idx, h)
        delta = .2*self.readout(y).tanh().reshape(-1, self.stages, c)
        features = []
        for stage, old in enumerate(state["features"]):
            patches = old[:, 1:].reshape(-1, c).clone().index_add(0, idx, delta[:, stage])
            features.append(torch.cat((old[:, :1], patches.reshape(b, n-1, c)), 1))
        return {**state, "features": features, "h": hidden}, {"updated_patches": len(idx)}
