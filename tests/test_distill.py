import torch

from sokkanaem.distill import distill_loss


def test_distill_loss_zero_when_aligned():
    B, N, dim, D = 2, 9, 8, 16
    proj = torch.nn.Linear(dim, D)
    tokens = torch.randn(B, N, dim)
    with torch.no_grad():
        target = proj(tokens).reshape(B, 3, 3, D)  # same direction as proj(tokens)
    loss = distill_loss(tokens, proj, target)
    assert loss.item() < 1e-5


def test_distill_loss_positive_when_misaligned():
    B, N, dim, D = 2, 9, 8, 16
    proj = torch.nn.Linear(dim, D)
    tokens = torch.randn(B, N, dim)
    target = torch.randn(B, 3, 3, D)  # unrelated direction
    loss = distill_loss(tokens, proj, target)
    assert loss.item() > 0.1
