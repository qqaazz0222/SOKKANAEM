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
    """In-frame context mixing. Full compute in PoC.
    # ponytail: static-patch output caching (gather-compute-scatter) is
    # phase-3 kernel work; here spatial is O(N) and not the bottleneck.
    """

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = BiSpatialSSM(dim, d_state)

    def forward(self, tokens):
        return tokens + self.ssm(self.norm(tokens))


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
                 gmc=False, gmc_lowres=128, gmc_corners=50):
        """gmc=True enables the ego-motion path (IDEA.md §3.5): Low-Res GMC
        warps frame t-1 onto t, then the change score is the relative L1
        between the *embed features* of both — not pixel MSE — so tau_on/
        tau_off are on the feature scale (see configs)."""
        super().__init__()
        self.p = patch_size
        self.dim = dim
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

    def _forward_tokens(self, tokens, mask, hs):
        """Run backbone. hs: list of temporal states (one per TemporalBlock)."""
        new_hs, ti = [], 0
        for blk in self.blocks:
            if isinstance(blk, TemporalBlock):
                tokens, h = blk.step(tokens, mask, hs[ti] if hs else None)
                new_hs.append(h)
                ti += 1
            else:
                tokens = blk(tokens)
        return tokens, new_hs

    def step(self, frame, state=None):
        """One streaming inference/training step.
        frame: (B, 3, H, W) in [0,1]. state: dict or None (stream start).
        All per-stream state (SSM hidden, detector bookkeeping, prev frame)
        lives in the dict — one model can serve interleaved streams.
        Returns depth (B,1,H,W), state, info dict."""
        B, _, H, W = frame.shape
        gh, gw = H // self.p, W // self.p
        if state is None:
            state = {"hs": None, "prev": None, "det": None}

        tokens = self.embed(frame).flatten(2).transpose(1, 2)  # (B, N, D)
        # ponytail: embedding of static patches is recomputed here; caching it
        # is a trivial win for the phase-3 kernel, irrelevant to PoC accuracy.
        if self.gmc is not None:
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
        tokens, hs = self._forward_tokens(tokens, mask, state["hs"])

        feat2d = tokens.transpose(1, 2).reshape(B, self.dim, gh, gw)
        depth = self.decoder(feat2d)
        info = {"mask": mask, "active_ratio": mask.mean().item()}
        return depth, {"hs": hs, "det": det,
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
                tokens, state["hs"] = self._forward_tokens(tokens, mask, state["hs"])
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
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model
