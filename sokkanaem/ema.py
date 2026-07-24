"""EMA shadow weights for eval-time self-ensembling.

Zero inference cost: the EMA copy is the same architecture, just swapped
in for eval/inference instead of the raw training weights (from_checkpoint
prefers it automatically). Only training pays the per-step update cost.
"""
import torch


def ema_update_(ema_state, model_state, decay):
    with torch.no_grad():
        for k, v in ema_state.items():
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(model_state[k], alpha=1 - decay)
            else:
                v.copy_(model_state[k])
