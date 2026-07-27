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


def test_multiscale_grad_loss_prefers_sharp_over_blurry():
    """The point of the pyramid term: a blurred prediction must be penalized
    even where its pixel-level gradients are small."""
    import torch.nn.functional as F

    from sokkanaem.losses import multiscale_grad_loss

    gt = torch.ones(1, 1, 64, 64) * 5.0
    gt[..., :32] = 2.0            # a depth step down the middle
    valid = torch.ones_like(gt)
    blur = F.avg_pool2d(gt, 9, stride=1, padding=4)

    sharp_l = multiscale_grad_loss(gt.clone(), gt, valid)
    blur_l = multiscale_grad_loss(blur, gt, valid)
    assert sharp_l < 1e-6, sharp_l
    assert blur_l > sharp_l, (blur_l, sharp_l)


def test_multiscale_grad_loss_scale_invariant():
    from sokkanaem.losses import multiscale_grad_loss
    gt = torch.rand(1, 1, 64, 64) * 8 + 1
    valid = torch.ones_like(gt)
    a = multiscale_grad_loss(gt * 3.0, gt, valid)
    assert a < 1e-4, "normalized disparity must absorb a global scale factor"
