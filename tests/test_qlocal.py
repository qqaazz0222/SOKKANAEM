import torch
from sokkanaem.qlocal import LocalRefiner, editable_mask
from sokkanaem.qseparated import SeparatedDelta


def test_edge_protection_and_bound():
    rgb = torch.zeros(1, 3, 32, 32)
    depth = torch.ones(1, 1, 32, 32); depth[..., 16:] = 2
    m = LocalRefiner(True)
    with torch.no_grad(): m.net[-1].bias.fill_(2.)
    mask = editable_mask(rgb, depth); out = m(rgb, depth)
    assert mask.any() and (~mask).any()
    assert torch.equal(out[~mask], depth[~mask])
    assert ((out/depth).log().abs() <= .030001).all()
    out.mean().backward(); assert m.net[-1].weight.grad.abs().sum() > 0


def test_rgb_boundary_protected_even_when_depth_flat():
    rgb = torch.zeros(1, 3, 32, 32); rgb[..., 16:] = 1
    depth = torch.ones(1, 1, 32, 32)
    mask = editable_mask(rgb, depth)
    assert not mask[..., 13:18].any()
    assert mask[..., :10].all()


def test_separated_delta_preserves_selected_global_inputs():
    torch.manual_seed(7)
    m = SeparatedDelta(True, True)
    old = m.refresh({"features": [torch.rand(1, 5, 384) for _ in range(4)],
                     "pooled": torch.rand(1, 384), "grid": (2, 2), "size": (28, 28)})
    with torch.no_grad():
        m.readout.bias.fill_(1.); m.global_readout.bias.fill_(1.)
    out, _ = m.update(torch.rand(1, 3, 28, 28), torch.ones(1, 4), old)
    assert torch.equal(out["pooled"], old["pooled"])
    for a, b in zip(out["features"], old["features"]):
        assert torch.equal(a[:, :1], b[:, :1]) and not torch.equal(a[:, 1:], b[:, 1:])
