import torch

from sokkanaem.losses import normal_loss


def test_normal_loss_zero_when_identical():
    y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    d = (0.1 * x + 0.2 * y).float()[None, None, None]  # (1,1,1,H,W) linear ramp
    valid = torch.ones_like(d)
    assert normal_loss(d, d, valid).item() < 1e-5


def test_normal_loss_positive_when_different():
    y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    ramp = (0.5 * x).float()[None, None, None]
    flat = torch.zeros_like(ramp)
    valid = torch.ones_like(ramp)
    assert normal_loss(ramp, flat, valid).item() > 0.01
