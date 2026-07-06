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
                 tau_on=0.02, tau_off=0.01, keyframe_every=30):
        super().__init__()
        self.p = patch_size
        self.dim = dim
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
        frame: (B, 3, H, W) in [0,1]. state: dict or None (resets detector).
        Returns depth (B,1,H,W), state, info dict."""
        B, _, H, W = frame.shape
        gh, gw = H // self.p, W // self.p
        if state is None:
            self.detector.reset()
            state = {"hs": None}

        mask = self.detector(frame)                       # (B, N)
        tokens = self.embed(frame).flatten(2).transpose(1, 2)  # (B, N, D)
        # ponytail: embedding of static patches is recomputed here; caching it
        # is a trivial win for the phase-3 kernel, irrelevant to PoC accuracy.
        tokens, hs = self._forward_tokens(tokens, mask, state["hs"])

        feat2d = tokens.transpose(1, 2).reshape(B, self.dim, gh, gw)
        depth = self.decoder(feat2d)
        info = {"mask": mask, "active_ratio": mask.mean().item()}
        return depth, {"hs": hs}, info

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
