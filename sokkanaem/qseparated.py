"""Post-training CLS versus metric-pooling delta removal, not a retrained model."""
import torch
from sokkanaem.qdelta_ssm import QDeltaSSM


class SeparatedDelta(QDeltaSSM):
    def __init__(self, hold_cls=False, hold_pooled=False):
        super().__init__()
        self.hold_cls = hold_cls; self.hold_pooled = hold_pooled

    def update(self, frame, mask, state):
        out, info = super().update(frame, mask, state)
        if self.hold_cls:
            out = {**out, "features": [torch.cat((old[:, :1], new[:, 1:]), 1)
                                        for old, new in zip(state["features"], out["features"])]}
        if self.hold_pooled:
            out = {**out, "pooled": state["pooled"]}
        return out, info
