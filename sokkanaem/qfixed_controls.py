"""Minimal periodic whole-output hold control for the fixed-camera study."""
import torch
from torch import nn


class PeriodicOutputHold(nn.Module):
    """No RGB detector, optical flow, interpolation, or learned correction."""
    def __init__(self, exact, period=2):
        super().__init__()
        if not isinstance(period, int) or period < 1:
            raise ValueError('Positive integer period required')
        self.exact = exact
        self.period = period

    @torch.no_grad()
    def step(self, frame, state=None):
        if frame.ndim != 4 or frame.shape[0] != 1:
            raise ValueError('One frame required')
        reset = state is None or state['depth'].shape[-2:] != frame.shape[-2:]
        refresh = reset or state['age'] + 1 >= self.period
        if refresh:
            depth, _, _ = self.exact.step(frame)
            age = 0
        else:
            depth = state['depth'].clone()
            age = state['age'] + 1
        return depth, {'depth': depth.detach().clone(), 'age': age}, {
            'full_refresh': refresh, 'backbone_calls': int(refresh),
            'cache_age': age, 'flow_calls': 0, 'mode': 'periodic_whole_output_hold_control'}
