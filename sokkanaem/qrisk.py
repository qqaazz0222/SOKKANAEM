"""Small learned refresh risk proxy. NOT calibrated uncertainty or a guarantee."""
import torch
from torch import nn
import torch.nn.functional as F
from scripts.qquality_study import MotionStream


def risk_features(frame, anchor, age):
    a = F.interpolate(anchor.float(), (37, 37), mode="bilinear", align_corners=False)
    b = F.interpolate(frame.float(), (37, 37), mode="bilinear", align_corners=False)
    d = b-a
    values = [d.abs().mean((-2, -1)), d.square().mean((-2, -1)).sqrt(),
              b.mean((-2, -1)), b.std((-2, -1), unbiased=False),
              b.new_full((1, 1), age/3), torch.diff(d, dim=-1).abs().mean((1, 2, 3))[:, None]]
    return torch.cat(values, 1)


class RefreshRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(14, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))
    def forward(self, x): return .05*F.softplus(self.net(x))


class RiskStream(MotionStream):
    def __init__(self, exact, adapter, risk, threshold):
        super().__init__(exact, adapter, refresh_every=4)
        self.risk = risk; self.threshold = threshold

    @torch.no_grad()
    def step(self, frame, state=None):
        predicted = 0.
        if state is not None and state["age"] < self.refresh_every-1:
            predicted = float(self.risk(risk_features(frame, state["anchor"], state["age"]+1)))
            if predicted > self.threshold:
                state = {**state, "age": self.refresh_every-1}
        pred, state, info = super().step(frame, state)
        info["predicted_risk"] = predicted
        return pred, state, info
