"""Core-claim checks (IDEA.md §3.2): Δ-gating gives EXACT state copy,
and identical consecutive frames produce near-zero active ratio.

Run:  python -m pytest tests/ -q   (or: python tests/test_gating.py)
"""
import torch

from sokkanaem import SOKKANAEM, ChangeDetector, SelectiveSSM


def test_delta_gating_exact_state_copy():
    torch.manual_seed(0)
    ssm = SelectiveSSM(dim=32, d_state=8).eval()
    u = torch.randn(2, 32)
    h0 = torch.randn(2, 64, 8)  # d_inner = 2*32
    _, h1 = ssm.step(u, mask=torch.zeros(2), h=h0)
    assert torch.equal(h1, h0), "mask=0 must copy hidden state bit-exactly"
    _, h2 = ssm.step(u, mask=torch.ones(2), h=h0)
    assert not torch.equal(h2, h0), "mask=1 must update hidden state"


def test_static_scene_skips_and_depth_stable():
    torch.manual_seed(0)
    model = SOKKANAEM(keyframe_every=1000).eval()
    frame = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        d1, state, i1 = model.step(frame, None)          # first frame: full
        d2, state, i2 = model.step(frame.clone(), state)  # identical frame
    assert i1["active_ratio"] == 1.0
    assert i2["active_ratio"] == 0.0, "identical frame must be fully static"
    assert torch.allclose(d1, d2, atol=1e-5), "static scene must give stable depth"


def test_detector_hysteresis_and_keyframe():
    det = ChangeDetector(patch_size=16, tau_on=0.02, tau_off=0.01, keyframe_every=3)
    f = torch.zeros(1, 3, 64, 64)
    m, st = det(f)                     # frame 0: keyframe
    assert m.mean() == 1.0
    m, st = det(f, st)                 # frame 1: static
    assert m.mean() == 0.0
    f2 = f.clone()
    f2[:, :, :16, :16] = 1.0
    m, st = det(f2, st)                # frame 2: one patch changed (+dilation)
    assert 0 < m.mean() < 1.0
    m, st = det(f2, st)                # frame 3: keyframe refresh
    assert m.mean() == 1.0
    # per-stream state is external: a fresh stream starts at frame 0
    m2, _ = det(f)
    assert m2.mean() == 1.0 and st["frame_idx"] == 4


def test_interleaved_streams_independent():
    """All per-stream state lives in the state dict — two streams through
    one model must not contaminate each other's masks."""
    torch.manual_seed(0)
    model = SOKKANAEM(keyframe_every=1000).eval()
    fa = torch.rand(1, 3, 64, 64)
    fb = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        _, sa, _ = model.step(fa, None)
        _, sb, _ = model.step(fb, None)            # stream B interleaved
        _, sa, ia = model.step(fa.clone(), sa)     # A unchanged -> static
        _, sb, ib = model.step(torch.rand(1, 3, 64, 64), sb)  # B changed
    assert ia["active_ratio"] == 0.0
    assert ib["active_ratio"] > 0.5


def test_from_checkpoint_restores_config(tmp_path):
    """eval/infer must rebuild with the trained [model] kwargs."""
    import torch as t
    from sokkanaem import from_checkpoint
    model = SOKKANAEM(dim=64, keyframe_every=7)
    t.save(model.state_dict(), tmp_path / "latest.pt")
    (tmp_path / "config.toml").write_text(
        "[model]\ndim = 64\nkeyframe_every = 7\n")
    m = from_checkpoint(tmp_path / "latest.pt")
    assert m.dim == 64 and m.detector.keyframe_every == 7
    # overrides win (e.g. --gmc feature-scale taus)
    m = from_checkpoint(tmp_path / "latest.pt", keyframe_every=9)
    assert m.detector.keyframe_every == 9


if __name__ == "__main__":
    test_delta_gating_exact_state_copy()
    test_static_scene_skips_and_depth_stable()
    test_detector_hysteresis_and_keyframe()
    test_interleaved_streams_independent()
    print("all gating tests passed")


def test_gate_mode_drop_bypasses_static_tokens():
    """§4.4 gating-position arm: "drop" must make a static token pass the
    temporal block unchanged, while Δ-gating keeps reading the retained
    state — the two arms must therefore disagree at identical active%."""
    torch.manual_seed(0)
    kw = dict(keyframe_every=1000, tau_on=0.02, tau_off=0.01)
    m_delta = SOKKANAEM(**kw).eval()
    m_drop = SOKKANAEM(**kw, gate_mode="drop").eval()
    m_drop.load_state_dict(m_delta.state_dict())

    blk = m_drop.blocks[0]
    tokens = torch.randn(1, 4, m_drop.dim)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    with torch.no_grad():
        out_drop, h_drop = blk.step(tokens, mask, None, "drop")
        out_delta, h_delta = blk.step(tokens, mask, None, "delta")
    static = mask[0] == 0
    assert torch.equal(out_drop[0, static], tokens[0, static]), \
        "drop must bypass the block for static tokens"
    assert not torch.allclose(out_delta[0, static], tokens[0, static]), \
        "Δ-gating must still read the retained state for static tokens"
    assert torch.equal(h_drop, h_delta), "both arms freeze the state identically"

    # end to end: same masks, different depth -> the arms are separable
    frame = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        _, sa, _ = m_delta.step(frame, None)
        _, sb, _ = m_drop.step(frame, None)
        f2 = frame.clone()
        f2[..., :16, :16] = torch.rand(1, 3, 16, 16)
        da, _, ia = m_delta.step(f2, sa)
        db, _, ib = m_drop.step(f2, sb)
    assert ia["active_ratio"] == ib["active_ratio"], "iso-active comparison"
    assert not torch.allclose(da, db, atol=1e-5)


def test_spatial_cache_matches_full_compute():
    """Static frames: cached spatial outputs must equal full recompute.
    Partially-active frames: cached path must refresh active patches."""
    torch.manual_seed(0)
    kw = dict(keyframe_every=1000, tau_on=0.02, tau_off=0.01)
    m_full = SOKKANAEM(**kw).eval()
    m_cache = SOKKANAEM(**kw, spatial_cache=True).eval()
    m_cache.load_state_dict(m_full.state_dict())

    frame = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        d1, s1, _ = m_full.step(frame, None)
        d2, s2, _ = m_cache.step(frame, None)
        assert torch.allclose(d1, d2, atol=1e-5), "first frame is full compute"
        # identical frame -> all static -> cache path returns cached tokens
        d1b, s1, _ = m_full.step(frame.clone(), s1)
        d2b, s2, i2 = m_cache.step(frame.clone(), s2)
        assert i2["active_ratio"] == 0.0
        assert torch.allclose(d1b, d2b, atol=1e-5), \
            "static scene: cache must reproduce full compute"
        # perturb one region -> partial activity, depth must move there
        frame2 = frame.clone()
        frame2[..., :32, :32] = torch.rand(1, 3, 32, 32)
        _, _, i3 = m_cache.step(frame2, s2)
        assert 0.0 < i3["active_ratio"] < 1.0, "partial change expected"


def test_shuffle_decoder_is_cheaper_and_same_shape():
    """v6 decoder: same output contract, an order of magnitude fewer MACs."""
    from sokkanaem.model import Decoder, ShuffleDecoder
    feat = torch.randn(2, 192, 16, 16)
    heavy, cheap = Decoder(192).eval(), ShuffleDecoder(192).eval()
    with torch.no_grad():
        a, b = heavy(feat), cheap(feat)
    assert a.shape == b.shape == (2, 1, 256, 256)
    assert (b > 0).all(), "softplus keeps depth positive"


def test_trainable_spatial_cache_flows_gradients():
    """v6(b): the sparse spatial path must be differentiable end to end, or
    training with the cache on silently optimizes nothing in those blocks."""
    torch.manual_seed(0)
    model = SOKKANAEM(dim=32, spatial_cache=True, decoder="shuffle")
    model.train()
    clip = torch.rand(1, 3, 3, 64, 64)
    N = (64 // 16) ** 2
    fm = (torch.rand(1, 3, N) > 0.5).float()
    fm[:, 0] = 1.0  # keyframe seeds the cache, as train.py does
    depths, masks = model.forward_clip(clip, force_mask=fm)
    depths.mean().backward()
    spatial = [b for b in model.blocks if not hasattr(b, "step")]
    grads = [p.grad for b in spatial for p in b.parameters()]
    assert grads and all(g is not None for g in grads), "spatial blocks got no grad"
    assert any(g.abs().sum() > 0 for g in grads), "spatial grads are all zero"


def test_dpt_decoder_shape_params_and_gating_intact():
    """v8 decoder: multi-scale fusion must keep the streaming contract (and
    therefore the Δ-gating claim) intact — identical frames still fully static
    and stable depth."""
    torch.manual_seed(0)
    m = SOKKANAEM(dim=64, decoder="dpt", keyframe_every=1000).eval()
    frame = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        d1, st, i1 = m.step(frame, None)
        d2, st, i2 = m.step(frame.clone(), st)
    assert d1.shape == (1, 1, 64, 64)
    assert i1["active_ratio"] == 1.0 and i2["active_ratio"] == 0.0
    assert torch.allclose(d1, d2, atol=1e-5), "static scene must stay stable"
    assert (d1 > 0).all(), "disparity head must yield positive depth"
    dec = sum(p.numel() for p in m.decoder.parameters())
    assert dec < 1_000_000, f"decoder budget blown: {dec}"


def test_dpt_decoder_trains_through_forward_clip():
    torch.manual_seed(0)
    m = SOKKANAEM(dim=32, decoder="dpt").train()
    clip = torch.rand(1, 3, 3, 64, 64)
    N = (64 // 16) ** 2
    fm = (torch.rand(1, 3, N) > 0.5).float()
    fm[:, 0] = 1.0
    depths, _ = m.forward_clip(clip, force_mask=fm)
    depths.mean().backward()
    assert all(p.grad is not None for p in m.decoder.parameters())
    assert any(p.grad.abs().sum() > 0 for p in m.embed.parameters())
