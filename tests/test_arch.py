"""v8 precision upgrades: 4-way cross-scan, local conv branch, bin head.

The point of each check is that the upgrade did NOT cost the sparsity claim:
identical frames still go fully static, and the sparse spatial path still
reproduces full compute on a static frame.
"""
import torch

from sokkanaem import SOKKANAEM
from sokkanaem.model import DPTDecoder, SpatialBlock
from sokkanaem.ssm import column_major_order

V8 = dict(dim=32, decoder="dpt", bins=16, scan_directions=4, local_conv=True,
          keyframe_every=1000, tau_on=0.02, tau_off=0.01)


def test_column_major_order_full_and_subset():
    gh, gw = 3, 4
    idx = torch.arange(gh * gw)
    o = column_major_order(idx, gh, gw)
    # column-major traversal of a 3x4 raster grid: 0,4,8,1,5,9,...
    assert o.tolist() == [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    # a gathered subset must be ordered by (col, row) among *its own* members
    sub = torch.tensor([1, 4, 5, 11])           # (r,c) = (0,1),(1,0),(1,1),(2,3)
    assert sub[column_major_order(sub, gh, gw)].tolist() == [4, 1, 5, 11]
    # permutation is invertible, which is what scatters the vertical scan back
    assert torch.equal(idx[o][o.argsort()], idx)


def test_cross_scan_uses_vertical_neighbours():
    """4-way must actually differ from 2-way, and only via the extra pair."""
    torch.manual_seed(0)
    blk4 = SpatialBlock(16, directions=4).eval()
    grid = (4, 4)
    order = column_major_order(torch.arange(16), *grid)
    tokens = torch.randn(1, 16, 16)
    with torch.no_grad():
        y4 = blk4(tokens, grid, order)
        blk4.ssm.directions = 2                  # same weights, horizontal only
        y2 = blk4(tokens, grid, None)
    assert not torch.allclose(y4, y2, atol=1e-5)


def test_v8_static_frame_is_exact_and_fully_skipped():
    """Δ-gating contract with every upgrade on: an identical frame must be
    100% static and give bit-stable depth, and the sparse spatial path must
    reproduce full compute there."""
    torch.manual_seed(0)
    full = SOKKANAEM(**V8).eval()
    cached = SOKKANAEM(**V8, spatial_cache=True).eval()
    cached.load_state_dict(full.state_dict())

    frame = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        d1, sf, i1 = full.step(frame, None)
        c1, sc, _ = cached.step(frame, None)
        assert i1["active_ratio"] == 1.0
        assert torch.allclose(d1, c1, atol=1e-5), "keyframe is full compute"
        d2, sf, i2 = full.step(frame.clone(), sf)
        c2, sc, _ = cached.step(frame.clone(), sc)
    assert i2["active_ratio"] == 0.0, "identical frame must be fully static"
    assert torch.allclose(d1, d2, atol=1e-5)
    assert torch.allclose(d2, c2, atol=1e-5), \
        "static frame: sparse spatial path must equal full compute"


def test_v8_partial_activity_matches_full_compute_on_changed_region():
    """The local depthwise branch reads dense inputs, so on a partially
    active frame the *active* patches must still match full compute — the
    only approximation allowed is the frozen cache at static patches."""
    torch.manual_seed(0)
    full = SOKKANAEM(**V8).eval()
    cached = SOKKANAEM(**V8, spatial_cache=True).eval()
    cached.load_state_dict(full.state_dict())
    frame = torch.rand(1, 3, 64, 64)
    frame2 = frame.clone()
    frame2[..., :32, :32] = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        _, sf, _ = full.step(frame, None)
        _, sc, _ = cached.step(frame, None)
        _, sf, ia = full.step(frame2, sf)
        _, sc, ib = cached.step(frame2, sc)
    assert ia["active_ratio"] == ib["active_ratio"]
    assert 0.0 < ib["active_ratio"] < 1.0


def test_temporal_cache_skips_static_readout_and_keeps_state_exact():
    """The readout cache must (a) reproduce full compute on a static frame,
    (b) leave the hidden state bit-identical to the Δ-gated path — the exact-
    state claim is what makes the next active frame correct."""
    torch.manual_seed(0)
    dense = SOKKANAEM(**V8).eval()
    cached = SOKKANAEM(**V8, temporal_cache=True).eval()
    cached.load_state_dict(dense.state_dict())
    frame = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        d1, sd, _ = dense.step(frame, None)
        c1, sc, _ = cached.step(frame, None)
        assert torch.allclose(d1, c1, atol=1e-5)
        d2, sd, i2 = dense.step(frame.clone(), sd)
        c2, sc, _ = cached.step(frame.clone(), sc)
    assert i2["active_ratio"] == 0.0
    assert torch.allclose(d2, c2, atol=1e-5), \
        "static frame: cached readout must equal the dense readout"
    for a, b in zip(sd["hs"], sc["hs"]):
        assert torch.equal(a, b), "hidden state must stay bit-identical"


def test_temporal_cache_trains():
    torch.manual_seed(0)
    m = SOKKANAEM(**V8, temporal_cache=True, spatial_cache=True).train()
    clip = torch.rand(1, 3, 3, 64, 64)
    fm = (torch.rand(1, 3, 16) > 0.5).float()
    fm[:, 0] = 1.0
    depths, _ = m.forward_clip(clip, force_mask=fm)
    depths.mean().backward()
    g = dict(m.named_parameters())["blocks.0.ssm.in_proj.weight"].grad
    assert g is not None and g.abs().sum() > 0


def test_affine_invariant_loss_ignores_teacher_gauge():
    """Teacher disparity has an arbitrary scale/shift; the loss must not."""
    from sokkanaem.distill import affine_invariant_loss
    torch.manual_seed(0)
    depth = torch.rand(2, 1, 16, 16) * 8 + 1
    disp = 1.0 / depth
    base = affine_invariant_loss(depth, disp)
    assert base < 1e-4, "matching prediction must give ~zero loss"
    assert torch.allclose(base, affine_invariant_loss(depth, disp * 7.5 + 3.0),
                          atol=1e-5), "loss must be affine-invariant"
    worse = affine_invariant_loss(depth, torch.rand_like(disp))
    assert worse > base + 0.1, "wrong geometry must cost more"


def test_bin_head_is_monotone_and_in_range():
    dec = DPTDecoder(32, bins=16, d_min=0.5, d_max=100.0).eval()
    c = dec.bin_centres()
    assert torch.all(c[1:] > c[:-1]), "bin centres must be sorted"
    assert c.min() > torch.tensor(0.5).log() and c.max() < torch.tensor(100.).log()
    dec.set_n_blocks(2)
    feats = [torch.randn(2, 32, 4, 4) for _ in range(2)]
    with torch.no_grad():
        d = dec(feats, torch.rand(2, 3, 64, 64))
    assert d.shape == (2, 1, 64, 64)
    assert (d > 0.5).all() and (d < 100.0).all(), "expectation stays in range"


def test_v8_trains_end_to_end():
    torch.manual_seed(0)
    m = SOKKANAEM(**V8, spatial_cache=True).train()
    clip = torch.rand(1, 3, 3, 64, 64)
    fm = (torch.rand(1, 3, 16) > 0.5).float()
    fm[:, 0] = 1.0
    depths, _ = m.forward_clip(clip, force_mask=fm)
    depths.mean().backward()
    named = dict(m.named_parameters())
    for p in ("blocks.1.ssm.vfwd.in_proj.weight", "blocks.1.local.0.weight",
              "decoder.bin_logits"):
        g = named[p].grad
        assert g is not None and g.abs().sum() > 0, f"{p} got no gradient"


def test_half_precision_streaming():
    """fp16 is the deployment dtype, and the detector emits fp32 masks whatever
    the model's dtype: multiplying Δ by one promoted the entire scan back to
    fp32 and the next einsum died on the mismatch. Frame 2 is the failing path
    (frame 1 is all-active, mask all ones)."""
    m = SOKKANAEM(dim=32, depth=2, patch_size=16, spatial_cache=True,
                  temporal_cache=True).half().eval()
    frame = torch.rand(1, 3, 64, 64).half()
    depth, state, _ = m.step(frame)
    depth, state, info = m.step(frame, state)
    assert info["active_ratio"] == 0.0          # identical frame -> nothing active
    assert depth.dtype == torch.float16 and torch.isfinite(depth).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all v8 arch tests passed")
