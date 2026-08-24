"""The fusion add must actually fuse: both branches on one scale.

DPTDecoder adds a backbone branch to an RGB-stem skip. On the reported
checkpoint the two arrived at wildly different magnitudes -- |red(x)| / |skip|
of 4.9, 6.8 and 9.8 at 1/8, 1/4 and 1/2 -- so the skip contributed nothing and
zeroing the whole stem barely moved the output. `fuse_norm=True` normalizes
each branch before the add.

The checks are about that imbalance, not about accuracy: with the norms on the
two branches must be comparable, and the stem must matter.
"""
import torch

from sokkanaem.model import DPTDecoder


def _branches(dec, frame, feats):
    """(|red(x)|, |skip|) per fusion level, as the decoder sees them."""
    import torch.nn.functional as F
    mags = []
    x = sum(p(f) for p, f in zip(dec.proj, feats))
    skips, h = [], frame
    for blk in dec.stem:
        h = blk(h)
        skips.append(h)
    for i, (skip, red, mix) in enumerate(
            zip(reversed(skips), dec.reduce, dec.mix)):
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)
        a, b = red(x), skip
        if dec.norm_x is not None:
            a, b = dec.norm_x[i](a), dec.norm_s[i](b)
        mags.append((a.abs().mean().item(), b.abs().mean().item()))
        x = mix(a + b)
    return mags


def _setup(fuse_norm):
    torch.manual_seed(0)
    dim, n = 32, 4
    dec = DPTDecoder(dim, 16, bins=8, fuse_norm=fuse_norm).eval()
    dec.set_n_blocks(n)
    frame = torch.rand(1, 3, 64, 64)
    # a backbone branch with the scale a trained one actually has: the sum
    # over blocks is what makes it large, so keep the sum
    feats = [torch.randn(1, dim, 4, 4) * 3 for _ in range(n)]
    return dec, frame, feats


def test_branches_are_comparable_with_norm():
    dec, frame, feats = _setup(True)
    for a, b in _branches(dec, frame, feats):
        assert 0.2 < a / b < 5.0, f"branch imbalance {a / b:.1f} survives the norm"


def _stem_influence(fuse_norm):
    """How much zeroing the RGB stem moves the prediction, relative."""
    dec, frame, feats = _setup(fuse_norm)
    with torch.no_grad():
        out = dec(feats, frame)
        hooks = [b.register_forward_hook(lambda m, i, o: o * 0)
                 for b in dec.stem]
        try:
            zeroed = dec(feats, frame)
        finally:
            for h in hooks:
                h.remove()
    return float(((out - zeroed).abs() / out.abs().clamp(min=1e-6)).mean())


def test_norm_gives_the_stem_a_voice():
    """The stem must matter more with the norms than without. This is the
    failure they exist to fix -- on the reported checkpoint zeroing the whole
    stem moved real AbsRel by 1.6% and improved the driving scene. An absolute
    threshold would only measure the random init, so the check is the
    comparison."""
    off, on = _stem_influence(False), _stem_influence(True)
    assert on > 3 * off, f"stem influence {off:.4f} -> {on:.4f}, not enough"


def test_off_by_default():
    dec, _, _ = _setup(False)
    assert dec.norm_x is None and dec.norm_s is None
