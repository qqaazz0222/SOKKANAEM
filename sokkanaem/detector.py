"""Patch-level change detection (IDEA.md §3.1).

Produces a binary active-patch mask from consecutive frames, with
hysteresis, dilation, and periodic keyframe refresh.
"""
import torch
import torch.nn.functional as F


class ChangeDetector:
    """Stateless math, stateful bookkeeping (prev frame, prev mask, counter).

    All ops are pixel-domain and negligible next to the backbone.
    """

    def __init__(self, patch_size=16, tau_on=0.02, tau_off=0.01,
                 dilate=True, keyframe_every=30):
        assert tau_on >= tau_off, "hysteresis requires tau_on >= tau_off"
        self.p = patch_size
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.dilate = dilate
        self.keyframe_every = keyframe_every
        self.reset()

    def reset(self):
        self.prev_frame = None
        self.prev_mask = None
        self.frame_idx = 0

    @torch.no_grad()
    def __call__(self, frame):
        """frame: (B, C, H, W) in [0, 1]. Returns mask (B, N) float 0/1,
        N = (H/p)*(W/p). First frame and keyframes are fully active."""
        B, C, H, W = frame.shape
        gh, gw = H // self.p, W // self.p

        keyframe = (self.prev_frame is None
                    or self.frame_idx % self.keyframe_every == 0)
        if keyframe:
            mask = torch.ones(B, gh * gw, device=frame.device)
        else:
            diff = (frame - self.prev_frame) ** 2  # (B, C, H, W)
            # mean squared diff per patch
            score = F.avg_pool2d(diff.mean(1, keepdim=True), self.p)  # (B,1,gh,gw)
            score = score[:, 0]
            prev = self.prev_mask.view(B, gh, gw)
            # hysteresis: active stays active until below tau_off
            m = torch.where(prev > 0.5, (score > self.tau_off), (score > self.tau_on)).float()
            if self.dilate:
                m = F.max_pool2d(m.unsqueeze(1), 3, stride=1, padding=1)[:, 0]
            mask = m.reshape(B, gh * gw)

        self.prev_frame = frame.clone()
        self.prev_mask = mask.clone()
        self.frame_idx += 1
        return mask
