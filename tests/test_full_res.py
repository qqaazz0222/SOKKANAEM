"""D1's learned upsampling must start as a no-op and then be able to do the one
thing bilinear cannot: put a depth step inside a 2x2 block."""
import torch

from sokkanaem.model import DPTDecoder


def _feats(dim=64, gh=8, n=4):
    torch.manual_seed(0)
    return [torch.randn(1, dim, gh, gh) for _ in range(n)], \
        torch.rand(1, 3, gh * 16, gh * 16)


def _decoder(full_res, dim=64, n=4, **kw):
    torch.manual_seed(0)
    d = DPTDecoder(dim, full_res=full_res, **kw)
    d.set_n_blocks(n)
    return d.eval()


def test_zero_init_residual_reproduces_the_bilinear_decoder():
    """A fresh D1 arm must emit exactly D0's depth: the residual is zero-init,
    so the arm can only be measured against a baseline it started equal to."""
    feats, frame = _feats()
    d0, d1 = _decoder(False), _decoder(True)
    # same weights for every shared module; the residual convs are extra
    missing, extra = d1.load_state_dict(d0.state_dict(), strict=False)
    assert not extra and all("up_" in k for k in missing)
    with torch.no_grad():
        a, b = d0(feats, frame), d1(feats, frame)
    assert torch.allclose(a, b, atol=1e-6)


def test_residual_can_place_a_step_inside_a_2x2_block():
    """Bilinear upsampling of a 1/2-res map has a two-pixel ramp at every edge.
    With the residual trained away from zero, odd and even columns move
    independently -- which is the whole point of the pixel-shuffle path."""
    feats, frame = _feats()
    d1 = _decoder(True)
    with torch.no_grad():
        base = d1(feats, frame)
        torch.nn.init.normal_(d1.up_res.weight, std=0.5)
        out = d1(feats, frame)
    assert not torch.allclose(base, out)
    # horizontal high-frequency energy, the component bilinear cannot carry
    def odd_even_gap(x):
        d = (x[..., 1:] - x[..., :-1]).abs()
        return float(d[..., ::2].mean() - d[..., 1::2].mean())
    assert abs(odd_even_gap(out)) > abs(odd_even_gap(base))


def test_bins_head_also_gets_the_residual():
    """The reported model runs bins=64, so a residual wired only into the
    scalar branch would be dead in every real run."""
    feats, frame = _feats()
    d = _decoder(True, bins=64)
    with torch.no_grad():
        base = d(feats, frame)
        torch.nn.init.normal_(d.up_res.weight, std=0.5)
        assert not torch.allclose(base, d(feats, frame))
