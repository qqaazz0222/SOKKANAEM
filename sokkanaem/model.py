"""SOKKANAEM model (IDEA.md §3.0–3.3).

Pipeline: patchify → change mask → interleaved T-Mamba / S-Mamba blocks
(Δ-gating on the temporal axis) → boundary-refinement decoder → dense depth.

Streaming API:
    state = None
    for frame in video:
        depth, state, info = model.step(frame, state)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .detector import ChangeDetector
from .gmc import GlobalMotionCompensator
from .ssm import SelectiveSSM, BiSpatialSSM


class TemporalBlock(nn.Module):
    """Per-patch-position scan across frames. Hidden state = visual memory
    of that location. Static patches copy state via Δ-gating (exact)."""

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = SelectiveSSM(dim, d_state)

    def step(self, tokens, mask, h):
        """tokens: (B, N, D), mask: (B, N), h: (B*N, d_inner, d_state) or None."""
        B, N, D = tokens.shape
        u = self.norm(tokens).reshape(B * N, D)   # each position = one stream
        y, h = self.ssm.step(u, mask.reshape(B * N), h)
        return tokens + y.reshape(B, N, D), h


class SpatialBlock(nn.Module):
    """In-frame context mixing. Full compute by default; with a cache of
    last outputs, active patches are gathered (raster order preserved),
    scanned as a subsequence, and scattered back — static patches reuse
    their cached output token (phase 3, IDEA.md §4.5). Approximation:
    the sparse scan skips static patches' context contribution; their
    context is frozen in the cache and refreshed on every keyframe."""

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = BiSpatialSSM(dim, d_state)

    def forward(self, tokens):
        if self.training and tokens.requires_grad:
            # chunked scan keeps (B, C, C, P, S) pairwise tensors alive for
            # backward — OOMs at 256px+. Recompute them instead.
            y = torch.utils.checkpoint.checkpoint(
                lambda t: self.ssm(self.norm(t)), tokens, use_reentrant=False)
            return tokens + y
        return tokens + self.ssm(self.norm(tokens))

    def forward_cached(self, tokens, mask, cache):
        """tokens: (B, N, D), mask: (B, N) 0/1, cache: (B, N, D) previous
        outputs. Returns (out, new_cache) — they are the same tensor."""
        if cache is None or bool(mask.all()):
            out = self.forward(tokens)
            return out, out
        out = cache.clone()
        for b in range(tokens.shape[0]):
            idx = mask[b].nonzero(as_tuple=True)[0]
            if idx.numel():
                sub = tokens[b:b + 1, idx]
                out[b, idx] = (sub + self.ssm(self.norm(sub)))[0]
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


class SOKKANAEM(nn.Module):
    def __init__(self, dim=192, depth=4, d_state=16, patch_size=16,
                 tau_on=0.02, tau_off=0.01, keyframe_every=30,
                 gmc=False, gmc_lowres=128, gmc_corners=50,
                 spatial_cache=False):
        """gmc=True enables the ego-motion path (IDEA.md §3.5): Low-Res GMC
        warps frame t-1 onto t, then the change score is the relative L1
        between the *embed features* of both — not pixel MSE — so tau_on/
        tau_off are on the feature scale (see configs)."""
        super().__init__()
        self.p = patch_size
        self.dim = dim
        self.spatial_cache = spatial_cache  # inference-only (§4.5 wall-clock)
        self._core = None  # CUDA-graph-captured full-compute path (opt-in)
        self.gmc = GlobalMotionCompensator(gmc_lowres, gmc_corners) if gmc else None
        self.embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        # interleave T, S, T, S, ...
        self.blocks = nn.ModuleList(
            TemporalBlock(dim, d_state) if i % 2 == 0 else SpatialBlock(dim, d_state)
            for i in range(depth)
        )
        self.decoder = Decoder(dim, patch_size)
        self.detector = ChangeDetector(patch_size, tau_on, tau_off,
                                       keyframe_every=keyframe_every)

    def _step_core(self, frame, mask, hs):
        """Full-compute streaming step as one pure-tensor function (embed →
        blocks → decoder). No CPU-dependent branching, so it is capturable
        by CUDA graphs via torch.compile(mode='reduce-overhead')."""
        B, _, H, W = frame.shape
        tokens = self.embed(frame).flatten(2).transpose(1, 2)
        tokens, new_hs, _ = self._forward_tokens(tokens, mask, hs)
        feat2d = tokens.transpose(1, 2).reshape(B, self.dim, H // self.p, W // self.p)
        return self.decoder(feat2d), new_hs

    def enable_cuda_graphs(self):
        """Route the full-compute path through CUDA graphs (inference only).
        The sparse spatial-cache path has dynamic shapes and stays eager —
        with spatial_cache=True this only accelerates keyframes' full pass
        never taken here, so it is a no-op; use on full-compute models."""
        self._core = torch.compile(self._step_core, mode="reduce-overhead")
        return self

    def _forward_tokens(self, tokens, mask, hs, sp=None):
        """Run backbone. hs: list of temporal states (one per TemporalBlock),
        sp: list of spatial output caches (one per SpatialBlock) or None
        (full spatial compute, no caching)."""
        new_hs, new_sp, ti, si = [], [], 0, 0
        for blk in self.blocks:
            if isinstance(blk, TemporalBlock):
                tokens, h = blk.step(tokens, mask, hs[ti] if hs else None)
                new_hs.append(h)
                ti += 1
            elif sp is not None:
                tokens, c = blk.forward_cached(tokens, mask, sp[si])
                new_sp.append(c)
                si += 1
            else:
                tokens = blk(tokens)
        return tokens, new_hs, new_sp

    def step(self, frame, state=None):
        """One streaming inference/training step.
        frame: (B, 3, H, W) in [0,1]. state: dict or None (stream start).
        All per-stream state (SSM hidden, detector bookkeeping, prev frame)
        lives in the dict — one model can serve interleaved streams.
        Returns depth (B,1,H,W), state, info dict."""
        B, _, H, W = frame.shape
        gh, gw = H // self.p, W // self.p
        if state is None:
            n_sp = sum(isinstance(b, SpatialBlock) for b in self.blocks)
            state = {"hs": None, "prev": None, "det": None, "sp": [None] * n_sp}

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

        if self._core is not None and not self.spatial_cache:
            depth, hs = self._core(frame, mask, state["hs"])
            # cudagraph-trees reuse output buffers across replays; detach
            # results from the static pool before they get overwritten
            depth = depth.clone()
            hs = [h.clone() for h in hs]
            sp = state["sp"]
        else:
            if tokens is None:
                tokens = self.embed(frame).flatten(2).transpose(1, 2)  # (B, N, D)
            tokens, hs, sp = self._forward_tokens(
                tokens, mask, state["hs"],
                sp=state["sp"] if self.spatial_cache else None)
            depth = self.decoder(
                tokens.transpose(1, 2).reshape(B, self.dim, gh, gw))
        info = {"mask": mask, "active_ratio": mask.mean().item()}
        return depth, {"hs": hs, "det": det, "sp": sp or state["sp"],
                       "prev": frame if self.gmc is not None else None}, info

    def forward_clip(self, clip, force_mask=None):
        """Training helper. clip: (B, T, 3, H, W). force_mask: (B, T, N) to
        override the detector (random-mask training, §3.2). Returns
        depths (B, T, 1, H, W), masks (B, T, N)."""
        B, T = clip.shape[:2]
        state, depths, masks = None, [], []
        for t in range(T):
            if force_mask is not None:
                # bypass detector, keep temporal state flowing
                if state is None:
                    state = {"hs": None}
                mask = force_mask[:, t]
                tokens = self.embed(clip[:, t]).flatten(2).transpose(1, 2)
                tokens, state["hs"], _ = self._forward_tokens(tokens, mask, state["hs"])
                gh, gw = clip.shape[-2] // self.p, clip.shape[-1] // self.p
                depth = self.decoder(tokens.transpose(1, 2).reshape(B, self.dim, gh, gw))
            else:
                depth, state, info = self.step(clip[:, t], state)
                mask = info["mask"]
            depths.append(depth)
            masks.append(mask)
        return torch.stack(depths, 1), torch.stack(masks, 1)


def from_checkpoint(ckpt, device="cpu", **overrides):
    """Rebuild the model with the [model] kwargs recorded next to the
    checkpoint (work_dirs/<name>/config.toml, saved by train.py), load
    weights, return it. overrides win — e.g. gmc=True with feature-scale
    taus replaces the trained pixel-scale ones."""
    import tomllib
    from pathlib import Path

    kw = {}
    cfg = Path(ckpt).parent / "config.toml"
    if cfg.exists():
        with open(cfg, "rb") as f:
            kw = tomllib.load(f).get("model", {})
    kw.update(overrides)
    model = SOKKANAEM(**kw).to(device)
    state = torch.load(ckpt, map_location=device)
    # ema (if present) is the eval-time shadow copy — prefer it over raw
    # training weights (self-ensembling, IDEA.md §6 style accuracy bump)
    model.load_state_dict(state.get("ema") or state.get("model") or state)
    return model
