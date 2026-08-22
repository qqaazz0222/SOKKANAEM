"""SOKKANAEM model (IDEA.md §3.0–3.3).

Pipeline: patchify → change mask → interleaved T-Mamba / S-Mamba blocks
(Δ-gating on the temporal axis) → boundary-refinement decoder → dense depth.

Streaming API:
    state = None
    for frame in video:
        depth, state, info = model.step(frame, state)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detector import ChangeDetector
from .gmc import GlobalMotionCompensator
from .ssm import SelectiveSSM, BiSpatialSSM, column_major_order


class TemporalBlock(nn.Module):
    """Per-patch-position scan across frames. Hidden state = visual memory
    of that location. Static patches copy state via Δ-gating (exact)."""

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = SelectiveSSM(dim, d_state)

    def step_cached(self, tokens, mask, h, cache, bucket=0):
        """Sparse counterpart of step(): only active positions are scanned.

        Δ-gating already makes the *state* update free for static patches, but
        the readout (in_proj/out_proj/C, 58.5% of an active token's MACs —
        scripts/flops.py) was still paid densely. A static patch's readout
        inputs barely move by construction (the detector said its pixels
        changed by less than tau), and its state is bit-identical, so its
        block output is reused from the previous frame instead. Same trade as
        the spatial cache: an approximation bounded by tau, refreshed on every
        keyframe, and trainable (train with temporal_cache on so the weights
        are fit to the path that runs at deployment).

        Returns (out, h, new_cache); out and new_cache are the same tensor."""
        B, N, D = tokens.shape
        if cache is None or h is None or bool(mask.all()):
            out, h = self.step(tokens, mask, h)
            return out, h, out
        flat = tokens.reshape(B * N, D)
        idx = mask.reshape(B * N).nonzero(as_tuple=True)[0]
        out = cache.clone()
        n = idx.numel()
        if n:
            if bucket > 0 and n % bucket:
                # every position here is an independent stream, so padding with
                # repeats of the last index needs no mask: the duplicates
                # recompute one stream and index_copy writes it the same value
                idx = torch.cat(
                    [idx, idx[-1:].expand(-(-n // bucket) * bucket - n)])
            y, h_act = self.ssm.step(self.norm(flat[idx]), None, h[idx])
            h = h.index_copy(0, idx, h_act)
            out = out.reshape(B * N, D).index_copy(
                0, idx, flat[idx] + y).reshape(B, N, D)
        return out, h, out

    def step(self, tokens, mask, h, gate_mode="delta"):
        """tokens: (B, N, D), mask: (B, N), h: (B*N, d_inner, d_state) or None.

        gate_mode="delta" (the contribution): Δ̃=0 freezes the state, and the
        block still *reads* that retained state with the current C — static
        patches keep contributing their accumulated temporal memory.
        gate_mode="drop": the IDEA.md §4.4 alternative — static tokens are
        dropped from the block entirely (identity bypass). State is frozen
        the same way, but the readout is gone, so the memory is unused.
        Isolates what the exact state *readout* is worth, at equal active%.
        """
        B, N, D = tokens.shape
        m = mask.reshape(B * N)
        u = self.norm(tokens).reshape(B * N, D)   # each position = one stream
        y, h = self.ssm.step(u, m, h)
        if gate_mode == "drop":
            y = y * m.unsqueeze(-1).to(y.dtype)
        return tokens + y.reshape(B, N, D), h


def pad_to_bucket(sub, order, bucket):
    """(1, n, D) gathered tokens -> (1, L, D) with L rounded up to a multiple
    of `bucket`, plus the Δ-gating mask that makes the pad free of side effects.

    The sparse path's whole problem as a *kernel* is that n changes every
    frame, so nothing can be captured in a CUDA graph and torch.compile
    re-traces (REPORT §4.19f: compiled dense already matches the sparse path
    at real-footage active rates). Padding to a handful of static shapes is
    the cheap half of the fix; a block-sparse Triton kernel is the other half.

    Exact, not an approximation: pads sit at the end with mask 0, so Δ=0
    freezes their state contribution in every scan direction — including the
    reversed one, where they are visited first and simply leave the initial
    state untouched. The pad tokens are copies of the last real token rather
    than zeros so nothing in the params path sees an out-of-distribution
    magnitude (their output is discarded either way).
    """
    n = sub.shape[1]
    if bucket <= 0:
        return sub, order, None
    L = -(-n // bucket) * bucket        # multiples, not powers of two: at 256
    if L == n:                          # tokens a bucket of 64 gives 4 shapes
        return sub, order, None         # and wastes <64 tokens, never 2x

    pad = L - n
    sub = torch.cat([sub, sub[:, -1:].expand(-1, pad, -1)], 1)
    if order is not None:
        tail = torch.arange(n, L, device=order.device)
        order = torch.cat([order, tail])
    m = torch.cat([sub.new_ones(1, n), sub.new_zeros(1, pad)], 1)
    return sub, order, m


class SpatialBlock(nn.Module):
    """In-frame context mixing. Full compute by default; with a cache of
    last outputs, active patches are gathered (raster order preserved),
    scanned as a subsequence, and scattered back — static patches reuse
    their cached output token (phase 3, IDEA.md §4.5). Approximation:
    the sparse scan skips static patches' context contribution; their
    context is frozen in the cache and refreshed on every keyframe.

    local_conv adds the CNN half of a CNN-Mamba hybrid: a depthwise 3x3 on
    the token grid, run *densely*. It stays exact under sparsity because its
    input (the temporal block's output) is dense — Δ-gating freezes state,
    not readout — so a neighbour's current value is always available; only
    the write is gated. Cost is dim*9 MAC/token, ~0.6% of one scan, so the
    compute-proportional-to-change-rate accounting barely moves."""

    def __init__(self, dim, d_state=16, directions=2, local_conv=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = BiSpatialSSM(dim, d_state, directions)
        self.local = None
        if local_conv:
            self.local_norm = nn.LayerNorm(dim)
            self.local = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.GELU(),
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim))

    def _mix(self, tokens, order, mask=None):
        if self.training and tokens.requires_grad:
            # chunked scan keeps (B, C, C, P, S) pairwise tensors alive for
            # backward — OOMs at 256px+. Recompute them instead.
            return torch.utils.checkpoint.checkpoint(
                lambda t: self.ssm(self.norm(t), order, mask), tokens,
                use_reentrant=False)
        return self.ssm(self.norm(tokens), order, mask)

    def _local(self, tokens, grid):
        """(B, N, D) -> (B, N, D) local 3x3 refinement, or None if disabled."""
        if self.local is None:
            return None
        B, N, D = tokens.shape
        gh, gw = grid
        x = self.local_norm(tokens).transpose(1, 2).reshape(B, D, gh, gw)
        return self.local(x).flatten(2).transpose(1, 2)

    def forward(self, tokens, grid, order=None):
        out = tokens + self._mix(tokens, order)
        loc = self._local(tokens, grid)
        return out if loc is None else out + loc

    def forward_cached(self, tokens, mask, cache, grid, order=None, bucket=0):
        """tokens: (B, N, D), mask: (B, N) 0/1, cache: (B, N, D) previous
        outputs. Returns (out, new_cache) — they are the same tensor.
        Differentiable, so this path can also be *trained* (v6): the cache
        stops being an untrained inference-time approximation.

        bucket>0 rounds the gathered length up to a power of two (>= bucket)
        with Δ-gated pad tokens, so the scan sees one of ~4 shapes instead of a
        new one every frame — see `pad_to_bucket`."""
        if cache is None or bool(mask.all()):
            out = self.forward(tokens, grid, order)
            return out, out
        loc = self._local(tokens, grid)   # dense but exact (inputs are dense)
        out = cache.clone()
        gh, gw = grid
        for b in range(tokens.shape[0]):
            idx = mask[b].nonzero(as_tuple=True)[0]
            n = idx.numel()
            if n:
                sub = tokens[b:b + 1, idx]
                # the gathered subsequence has its own column-major order
                sub_order = (column_major_order(idx, gh, gw)
                             if self.ssm.directions == 4 else None)
                sub, sub_order, m = pad_to_bucket(sub, sub_order, bucket)
                new = (sub + self._mix(sub, sub_order, m))[0, :n]
                out[b, idx] = new if loc is None else new + loc[b, idx]
        return out, out


class Decoder(nn.Module):
    """Lightweight upsampling decoder + boundary refinement convs (§3.3).
    Budget deliberately small relative to backbone."""

    def __init__(self, dim, patch_size=16):
        super().__init__()
        assert patch_size == 16
        self.net = nn.Sequential(
            nn.Conv2d(dim, 128, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, feat2d):
        return F.softplus(self.net(feat2d))  # positive depth


class ShuffleDecoder(nn.Module):
    """Same job as Decoder, ~10x fewer MACs (v6, REPORT.md §4.10).

    Decoder upsamples first and then runs 64->32 and 32->1 convs at FULL
    resolution: 1.59 of the model's 2.34 GMAC per 256px frame, i.e. 68% of
    the compute, versus the "decoder budget <= 10% of the backbone" that
    IDEA.md §3.3 set for itself. With a dense block that large, no amount
    of patch skipping can move total FLOPs (measured: 96.4% of full compute
    at active=0%). Here all channel work happens at patch resolution and
    pixel-shuffle does the upsampling, leaving one thin 1->16->1 pass at
    full resolution for the boundary refinement §3.3 asks for."""

    def __init__(self, dim, patch_size=16):
        super().__init__()
        self.p = patch_size
        self.head = nn.Sequential(
            nn.Conv2d(dim, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, patch_size * patch_size, 3, padding=1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, feat2d):
        x = F.pixel_shuffle(self.head(feat2d), self.p)
        return F.softplus(x + self.refine(x))  # residual: refine boundaries only


class DPTDecoder(nn.Module):
    """Multi-scale fusion decoder (DPT/MiDaS practice), ~0.6M params.

    Decoder and ShuffleDecoder both upsample ONE 16x16 feature map with no
    skip connection, so nothing in the network ever sees image detail finer
    than a patch. Every high-precision depth model instead fuses features at
    several resolutions. Here:
      * every backbone block's tokens are projected and summed (deep semantic
        features + shallow ones, the "reassemble" idea without extra scales
        inside the SSM, which runs at one resolution by construction);
      * a 3-layer stride-2 conv stem on the RGB frame supplies 1/8, 1/4 and
        1/2 detail features as skips;
      * fusion goes 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1, one 3x3 conv per level.

    Output is inverse depth (disparity) when disparity=True: 0.5-129 m ranges
    like TartanAir V2's compress far better in disparity space, and that is
    what MiDaS/DA v2 regress.

    bins > 0 switches the head to classification-regression (AdaBins/BinsFormer
    practice): the head emits `bins` logits per pixel, softmax over them, and
    the depth is their expectation over learned log-depth bin centres. A direct
    scalar regression head has to move one number across the whole range and
    ends up averaging across boundaries; a distribution can be bimodal at an
    occlusion edge and still resolve to a sharp value. Bin widths are global
    and learned (softmax over `bins` logits), not per-image — no extra
    inference cost, no extra transformer.

    Channels taper as resolution grows (16 at 1/2), and the last 2x upsample is
    bilinear on a 1-channel map: a 64-channel 3x3 conv at full 256px res alone
    would cost 2.4 GMAC, more than the whole rest of the model.
    """

    def __init__(self, dim, patch_size=16, width=64, disparity=True,
                 bins=0, d_min=0.3, d_max=150.0):
        super().__init__()
        assert patch_size == 16
        self.disparity = disparity
        self.bins = bins
        self.width, self.dim = width, dim
        self.proj = nn.ModuleList()  # per-block, set by set_n_blocks
        chans = (16, 32, width)      # stem output channels at 1/2, 1/4, 1/8
        self.stem = nn.ModuleList([
            nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.GELU())
            for cin, cout in zip((3,) + chans[:-1], chans)
        ])
        # fusion 1/16 -> 1/8 -> 1/4 -> 1/2, each: reduce, add skip, mix
        widths = (width, 32, 16)
        self.reduce = nn.ModuleList(
            nn.Conv2d(cin, cout, 1)
            for cin, cout in zip((width,) + widths[:-1], widths))
        self.mix = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GELU())
            for c in widths)
        self.head = nn.Sequential(
            nn.Conv2d(widths[-1], widths[-1], 3, padding=1), nn.GELU(),
            nn.Conv2d(widths[-1], max(bins, 1), 3, padding=1),
        )
        if bins:
            self.bin_logits = nn.Parameter(torch.zeros(bins))
            self.bin_temp = 1.0
            self.register_buffer(
                "log_range",
                torch.tensor([math.log(d_min), math.log(d_max)]))

    def bin_centres(self):
        """Learned log-depth bin centres, (bins,). Widths are a softmax so
        they stay positive and sum to the full log range — the bins can
        redistribute towards the depths a dataset actually contains, but can
        never collapse or leave the range."""
        lo, hi = self.log_range
        w = F.softmax(self.bin_logits, 0) * (hi - lo)
        return lo + torch.cumsum(w, 0) - w / 2

    def set_n_blocks(self, n):
        self.proj = nn.ModuleList(
            nn.Conv2d(self.dim, self.width, 1) for _ in range(n))

    def forward(self, feats2d, frame):
        """feats2d: list of (B, dim, gh, gw), one per backbone block.
        frame: (B, 3, H, W) — the RGB detail source."""
        x = sum(p(f) for p, f in zip(self.proj, feats2d))    # 1/16
        skips, h = [], frame
        for blk in self.stem:
            h = blk(h)
            skips.append(h)                                  # 1/2, 1/4, 1/8
        for skip, red, mix in zip(reversed(skips), self.reduce, self.mix):
            x = F.interpolate(x, scale_factor=2, mode="bilinear",
                              align_corners=False)
            x = mix(red(x) + skip)
        if self.bins:
            # expectation over bin centres, in log-depth, at 1/2 res; the
            # upsample then interpolates log-depth (geometric, smooth across
            # scale) and exp() keeps the output positive by construction.
            # bin_temp < 1 sharpens the distribution at inference. It exists
            # because "the expectation over bins pulls toward the mean" is a
            # testable explanation for range compression (Section 6.5), and the
            # test needs a knob rather than a patched forward pass.
            p = (self.head(x) / self.bin_temp).softmax(1)
            logd = (p * self.bin_centres().view(1, -1, 1, 1)).sum(1, keepdim=True)
            return F.interpolate(logd, scale_factor=2, mode="bilinear",
                                 align_corners=False).exp()
        out = F.interpolate(self.head(x), scale_factor=2, mode="bilinear",
                           align_corners=False)
        if self.disparity:
            # regress disparity (MiDaS/DA v2 practice): 0.5-129 m ranges like
            # TartanAir V2's are far better conditioned in inverse space.
            # Returned as metres so losses/eval stay unchanged.
            return 1.0 / (F.softplus(out) + 1e-3)
        return F.softplus(out)


class SOKKANAEM(nn.Module):
    def __init__(self, dim=192, depth=4, d_state=16, patch_size=16,
                 tau_on=0.02, tau_off=0.01, keyframe_every=30,
                 refresh="keyframe",
                 gmc=False, gmc_lowres=128, gmc_corners=50,
                 spatial_cache=False, gate_mode="delta", decoder="conv",
                 scan_directions=2, local_conv=False, bins=0,
                 temporal_cache=False, dense_above=0.4,
                 d_min=0.3, d_max=150.0, bucket=0):
        """gmc=True enables the ego-motion path (IDEA.md §3.5): Low-Res GMC
        warps frame t-1 onto t, then the change score is the relative L1
        between the *embed features* of both — not pixel MSE — so tau_on/
        tau_off are on the feature scale (see configs).

        scan_directions=4 / local_conv / bins are the v8 precision upgrades.
        All three sit in paths that either scale with active% (the spatial
        scan, under spatial_cache) or cost ~nothing dense (depthwise 3x3,
        bin logits at 1/2 res) — the point of SOKKANAEM is that unchanged
        regions do not pay, so extra capacity goes where sparsity applies,
        not into the dense decoder (REPORT.md §4.11)."""
        super().__init__()
        self.p = patch_size
        self.dim = dim
        self._order = {}  # grid -> dense column-major permutation (4-way scan)
        # Above this active ratio a frame takes the dense path instead, which
        # also refreshes every cache entry so the next sparse frame starts
        # clean. Measured on the 60k checkpoint (REPORT §4.20a): real-footage
        # AbsRel 0.1685 -> 0.1633 and delta1 0.8083 -> 0.8211, paid for with
        # active 22.2% -> 32.2%. (The original motivation, §3.3's U-shaped
        # t-delta blowup at 54-72% active, was a v6 artifact of an untrained
        # cache path and no longer reproduces — the gain that remains is
        # cutting stale context on medium-motion real footage.)
        self.dense_above = dense_above
        # round the gathered active-token count up to a multiple of this so the
        # sparse scan sees a handful of static shapes (0 = off, see
        # pad_to_bucket). Purely a kernel-shape knob: results are unchanged.
        self.bucket = bucket
        self.spatial_cache = spatial_cache  # inference-only (§4.5 wall-clock)
        self.temporal_cache = temporal_cache  # skip static tokens' readout too
        assert not (temporal_cache and gate_mode == "drop"), \
            "the drop ablation needs the dense readout path to compare against"
        self.gate_mode = gate_mode  # "delta" | "drop" — §4.4 gating-position abl.
        self._core = None  # CUDA-graph-captured full-compute path (opt-in)
        self.gmc = GlobalMotionCompensator(gmc_lowres, gmc_corners) if gmc else None
        self.embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        # interleave T, S, T, S, ...
        self.blocks = nn.ModuleList(
            TemporalBlock(dim, d_state) if i % 2 == 0
            else SpatialBlock(dim, d_state, scan_directions, local_conv)
            for i in range(depth)
        )
        # "conv" = the original (v1-v5 checkpoints); "shuffle" = v6 low-res;
        # "dpt" = v8 multi-scale fusion with RGB skips (needs block features
        # and the frame, so it takes a different call signature)
        self.decoder_kind = decoder
        if decoder == "dpt":
            # d_max is a real accuracy knob, not a formality: at 150 the top
            # bin centre sits at 115 m, and on vkitti2 the 0.8% of pixels past
            # that carry 54% of the squared error (scripts/bin_probe.py)
            self.decoder = DPTDecoder(dim, patch_size, bins=bins,
                                      d_min=d_min, d_max=d_max)
            self.decoder.set_n_blocks(depth)
        else:
            assert not bins, "bins head lives in the dpt decoder"
            self.decoder = (ShuffleDecoder if decoder == "shuffle" else Decoder)(
                dim, patch_size)
        self.detector = ChangeDetector(patch_size, tau_on, tau_off,
                                       keyframe_every=keyframe_every,
                                       refresh=refresh)
        self._n_spatial = sum(isinstance(b, SpatialBlock) for b in self.blocks)
        self._n_temporal = sum(isinstance(b, TemporalBlock) for b in self.blocks)

    def _step_core(self, frame, mask, hs):
        """Full-compute streaming step as one pure-tensor function (embed →
        blocks → decoder). No CPU-dependent branching, so it is capturable
        by CUDA graphs via torch.compile(mode='reduce-overhead')."""
        grid = (frame.shape[-2] // self.p, frame.shape[-1] // self.p)
        tokens = self.embed(frame).flatten(2).transpose(1, 2)
        tokens, new_hs, _, _, feats = self._forward_tokens(tokens, mask, hs, grid)
        return self._decode(tokens, feats, frame), new_hs

    def enable_cuda_graphs(self):
        """Route the full-compute path through CUDA graphs (inference only).
        The sparse spatial-cache path has dynamic shapes and stays eager —
        with spatial_cache=True this only accelerates keyframes' full pass
        never taken here, so it is a no-op; use on full-compute models."""
        self._core = torch.compile(self._step_core, mode="reduce-overhead")
        return self

    def compile_sparse(self):
        """Compile the sparse path's inner scans (inference only, needs
        bucket>0). The gather itself stays eager — `mask.nonzero()` has a
        data-dependent output shape and nothing can capture that — but with
        bucketing everything downstream of it sees one of a handful of static
        shapes, so each gets its own specialization instead of a re-trace per
        frame. Plain compile, not reduce-overhead: cudagraphs would hand back
        pooled output buffers that the caches then hold across replays."""
        assert self.bucket > 0, "compile_sparse without bucket>0 re-traces per frame"
        for blk in self.blocks:
            if isinstance(blk, SpatialBlock):
                blk._mix = torch.compile(blk._mix, dynamic=False)
            else:
                blk.ssm.step = torch.compile(blk.ssm.step, dynamic=False)
        return self

    def _dense_order(self, grid, device):
        """Column-major permutation of the full grid, cached per grid size
        (only needed by the 4-way cross-scan)."""
        key = (grid, str(device))
        if key not in self._order:
            n = grid[0] * grid[1]
            self._order[key] = column_major_order(
                torch.arange(n, device=device), *grid)
        return self._order[key]

    def _forward_tokens(self, tokens, mask, hs, grid, sp=None, tc=None):
        """Run backbone. hs: list of temporal states (one per TemporalBlock),
        grid: (gh, gw) patch-grid shape, sp: list of spatial output caches
        (one per SpatialBlock) or None (full spatial compute, no caching),
        tc: list of temporal output caches (one per TemporalBlock) or None."""
        new_hs, new_sp, new_tc, feats, ti, si = [], [], [], [], 0, 0
        cross = any(getattr(b, "ssm", None) is not None
                    and getattr(b.ssm, "directions", 2) == 4
                    for b in self.blocks if isinstance(b, SpatialBlock))
        order = self._dense_order(grid, tokens.device) if cross else None
        for blk in self.blocks:
            if isinstance(blk, TemporalBlock):
                h_in = hs[ti] if hs else None
                if tc is not None:
                    tokens, h, c = blk.step_cached(tokens, mask, h_in, tc[ti],
                                                   self.bucket)
                    new_tc.append(c)
                else:
                    tokens, h = blk.step(tokens, mask, h_in, self.gate_mode)
                new_hs.append(h)
                ti += 1
            elif sp is not None:
                tokens, c = blk.forward_cached(tokens, mask, sp[si], grid,
                                              order, self.bucket)
                new_sp.append(c)
                si += 1
            else:
                tokens = blk(tokens, grid, order)
            feats.append(tokens)   # per-block features for the DPT decoder
        return tokens, new_hs, new_sp, new_tc, feats

    def _decode(self, tokens, feats, frame):
        """Tokens -> dense depth, via whichever decoder this model was built
        with. The DPT decoder additionally consumes every block's features and
        the RGB frame (multi-scale fusion), so it needs its own call."""
        B, _, H, W = frame.shape
        gh, gw = H // self.p, W // self.p
        if self.decoder_kind == "dpt":
            return self.decoder(
                [t.transpose(1, 2).reshape(B, self.dim, gh, gw) for t in feats],
                frame)
        return self.decoder(tokens.transpose(1, 2).reshape(B, self.dim, gh, gw))

    def step(self, frame, state=None):
        """One streaming inference/training step.
        frame: (B, 3, H, W) in [0,1]. state: dict or None (stream start).
        All per-stream state (SSM hidden, detector bookkeeping, prev frame)
        lives in the dict — one model can serve interleaved streams.
        Returns depth (B,1,H,W), state, info dict."""
        B, _, H, W = frame.shape
        gh, gw = H // self.p, W // self.p
        if state is None:
            state = {"hs": None, "prev": None, "det": None,
                     "sp": [None] * self._n_spatial,
                     "tc": [None] * self._n_temporal}

        tokens = None
        # ponytail: embedding of static patches is recomputed here; caching it
        # is a trivial win for the phase-3 kernel, irrelevant to PoC accuracy.
        if self.gmc is not None:
            tokens = self.embed(frame).flatten(2).transpose(1, 2)  # (B, N, D)
            # §3.5: GMC-align prev frame, gate on embed-feature diff.
            # Keyframes are all-active regardless — skip the warp + embed.
            score = None
            if not self.detector.is_keyframe(state["det"]) \
                    and state["prev"] is not None:
                with torch.no_grad():
                    warped = self.gmc(state["prev"], frame)
                    fw = self.embed(warped).flatten(2).transpose(1, 2)
                    # relative L1 per patch — scale-free, robust to feature
                    # magnitude drift across training
                    num = (tokens - fw).abs().mean(-1)
                    den = tokens.abs().mean(-1) + fw.abs().mean(-1) + 1e-6
                    score = (num / den).view(B, gh, gw)
            mask, det = self.detector.gate(score, B, gh, gw,
                                           frame.device, state["det"])
        else:
            mask, det = self.detector(frame, state["det"])  # (B, N)

        if (self.spatial_cache or self.temporal_cache) and 0 < self.dense_above \
                and mask.mean() > self.dense_above:
            # too much motion for the caches to pay off — run the frame dense.
            # active_ratio reports 1.0 because that is the compute we paid.
            mask = torch.ones_like(mask)

        if self._core is not None and not self.spatial_cache:
            # populate the cross-scan permutation cache OUTSIDE the graph: a
            # tensor first created inside a captured region lives in the
            # cudagraph pool, and the next replay overwrites the entry this
            # dict still points at (4-way scan + --compile died on exactly that)
            self._dense_order((gh, gw), frame.device)
            depth, hs = self._core(frame, mask, state["hs"])
            # cudagraph-trees reuse output buffers across replays; detach
            # results from the static pool before they get overwritten
            depth = depth.clone()
            hs = [h.clone() for h in hs]
            sp, tc = state["sp"], state["tc"]
        else:
            if tokens is None:
                tokens = self.embed(frame).flatten(2).transpose(1, 2)  # (B, N, D)
            tokens, hs, sp, tc, feats = self._forward_tokens(
                tokens, mask, state["hs"], (gh, gw),
                sp=state["sp"] if self.spatial_cache else None,
                tc=state["tc"] if self.temporal_cache else None)
            depth = self._decode(tokens, feats, frame)
        info = {"mask": mask, "active_ratio": mask.mean().item()}
        return depth, {"hs": hs, "det": det, "sp": sp or state["sp"],
                       "tc": tc or state["tc"],
                       "prev": frame if self.gmc is not None else None}, info

    def forward_clip(self, clip, force_mask=None, return_tokens=False):
        """Training helper. clip: (B, T, 3, H, W). force_mask: (B, T, N) to
        override the detector (random-mask training, §3.2). return_tokens
        additionally returns pre-decoder backbone tokens (B, T, N, dim) —
        force_mask path only, for feature distillation (sokkanaem/distill.py).
        Returns depths (B, T, 1, H, W), masks (B, T, N)[, tokens]."""
        B, T = clip.shape[:2]
        state, depths, masks = None, [], []
        all_tokens = [] if return_tokens else None
        for t in range(T):
            if force_mask is not None:
                # bypass detector, keep temporal state flowing
                if state is None:
                    state = {"hs": None, "sp": [None] * self._n_spatial,
                             "tc": [None] * self._n_temporal}
                mask = force_mask[:, t]
                tokens = self.embed(clip[:, t]).flatten(2).transpose(1, 2)
                # v6: with spatial_cache the sparse spatial path is trained,
                # not bolted on at inference. force_mask[:, 0] is all-ones
                # (train.py), so the cache is always seeded by a full frame.
                grid = (clip.shape[-2] // self.p, clip.shape[-1] // self.p)
                tokens, state["hs"], sp, tc, feats = self._forward_tokens(
                    tokens, mask, state["hs"], grid,
                    sp=state["sp"] if self.spatial_cache else None,
                    tc=state["tc"] if self.temporal_cache else None)
                if self.spatial_cache:
                    state["sp"] = sp
                if self.temporal_cache:
                    state["tc"] = tc
                if return_tokens:
                    all_tokens.append(tokens)
                depth = self._decode(tokens, feats, clip[:, t])
            else:
                assert not return_tokens, "return_tokens needs force_mask (detector path has no token hook)"
                depth, state, info = self.step(clip[:, t], state)
                mask = info["mask"]
            depths.append(depth)
            masks.append(mask)
        out = torch.stack(depths, 1), torch.stack(masks, 1)
        return out + (torch.stack(all_tokens, 1),) if return_tokens else out


def checkpoint_config(ckpt):
    """The whole sidecar config next to a checkpoint (work_dirs/<name>/
    config.toml, written by train.py), or {} when absent.

    Callers need the top-level training args, not only [model]: `size`
    especially — inference at a resolution the model was never trained at is
    silent, and the 128 default used to override a 256px run."""
    from pathlib import Path

    # tomllib is 3.11+; a JetPack 4.6 image is Python 3.6, so fall back to the
    # backports rather than making the edge benchmark unrunnable
    try:
        import tomllib as _toml
        _binary = True
    except ImportError:
        try:
            import tomli as _toml
            _binary = True
        except ImportError:
            import toml as _toml          # pure-python, works on 3.6
            _binary = False

    cfg = Path(ckpt).parent / "config.toml"
    if not cfg.exists():
        return {}
    with open(cfg, "rb" if _binary else "r") as f:
        return _toml.load(f)


def from_checkpoint(ckpt, device="cpu", **overrides):
    """Rebuild the model with the [model] kwargs recorded next to the
    checkpoint (work_dirs/<name>/config.toml, saved by train.py), load
    weights, return it. overrides win — e.g. gmc=True with feature-scale
    taus replaces the trained pixel-scale ones."""
    kw = checkpoint_config(ckpt).get("model", {})
    kw.update(overrides)
    model = SOKKANAEM(**kw).to(device)
    state = torch.load(ckpt, map_location=device)
    # ema (if present) is the eval-time shadow copy — prefer it over raw
    # training weights (self-ensembling, IDEA.md §6 style accuracy bump)
    model.load_state_dict(state.get("ema") or state.get("model") or state)
    return model
