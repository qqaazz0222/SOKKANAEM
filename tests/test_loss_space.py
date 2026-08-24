"""The gradient and normal terms have to be able to see indoor footage.

Both were computed on raw metres in the reported recipe. Two consequences,
one per test here:

  * a pseudo-normal built from metric depth gradients is (0,0,1) wherever the
    surface is smooth and the room is small -- a wall receding across a 1.5-4 m
    indoor scene has dz/dx ~ 0.04 per pixel -- so `normal_loss`, the term whose
    job is surface slope, barely responds when that wall is predicted flatter
    than it is. Move the same scene out to 15-40 m, the same *relative* shape,
    and it responds a thousand times more strongly. (At a hard occlusion step
    the term saturates at both ends instead, which is why the check below uses
    a smooth ramp: that is where the signal was supposed to be.)
  * `multiscale_grad_loss` normalized disparity over the whole batch, and
    batches are drawn from a five-source mixture, so a 0.5-129 m synthetic
    frame sets the scale and the indoor frame in the same batch shrinks away.

`--loss-space log` fixes both. These check the responses, not loss values.
"""
import torch

from sokkanaem.losses import _norm_disp, normal_loss


def _ramp(lo, hi, size=64, n=1):
    """A receding surface: depth sweeps lo -> hi across the image, log-linearly
    so that two ramps with the same ratio have the same *shape*."""
    t = torch.linspace(0, 1, size)
    d = (torch.tensor(lo).log() + t * (torch.tensor(hi / lo).log())).exp()
    return d.view(1, 1, 1, size).expand(n, 1, size, size).contiguous()


def _flattened(d, factor=0.7):
    """The same surface predicted flatter -- the defect this paper calls range
    compression, here in its spatial form."""
    lg = d.clamp(min=1e-3).log()
    return (lg.mean() + (lg - lg.mean()) * factor).exp()


def test_normal_loss_is_blind_to_a_small_room_in_metres():
    """One shape, two absolute scales. 1.5-4 m and 15-40 m are the same ramp
    up to a factor of ten, so a scale-free term must score them alike."""
    def response(lo, hi):
        gt = _ramp(lo, hi)
        bad = _flattened(gt)
        v = torch.ones_like(gt)
        lg = lambda x: x.clamp(min=1e-3).log()
        return float(normal_loss(bad, gt, v)), float(normal_loss(lg(bad), lg(gt), v))

    near_m, near_log = response(1.5, 4.0)
    far_m, far_log = response(15.0, 40.0)
    # metric space: the identical shape defect is orders of magnitude louder
    # ten times further away, so the indoor sources barely reach the term
    assert far_m > 20 * near_m, (near_m, far_m)     # measured ~74x
    # log space: scale-free, so both report the same defect
    assert 0.5 < near_log / far_log < 2.0, (near_log, far_log)


def _mad(x, v):
    """Mean absolute deviation from the median, per frame -- the thing
    `_norm_disp` is supposed to set to 1."""
    return [float((x[i][v[i].bool()] - x[i][v[i].bool()].median()).abs().mean())
            for i in range(x.shape[0])]


def test_per_sample_norm_puts_every_frame_on_its_own_scale():
    """MiDaS normalizes disparity per image. Normalizing per batch instead
    leaves a frame whose range is narrow relative to its batch-mates scaled
    down by that ratio, and every gradient computed on it with it -- so the
    term that exists to supply sharpness is quietly down-weighted on exactly
    the frames it matters for.

    Measured on real mixed batches (five sources, batch 4, clip 4): frames with
    a per-frame scale of 0.039 shared a batch scale of 0.543, i.e. a 14x
    down-weighting. This checks the invariant behind that number."""
    gt = torch.cat([_ramp(1.5, 4.0), _ramp(0.5, 129.0)], 0)
    valid = torch.ones_like(gt)

    per = _mad(_norm_disp(gt, valid, per_sample=True), valid)
    batch = _mad(_norm_disp(gt, valid, per_sample=False), valid)

    for m in per:                       # every frame ends at unit scale
        assert abs(m - 1.0) < 0.01, per
    assert min(batch) < 0.5, batch      # under one shared scale, one does not
