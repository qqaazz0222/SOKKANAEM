"""Patch-level change detection (IDEA.md §3.1).

Produces a binary active-patch mask from consecutive frames, with
hysteresis, dilation, and periodic keyframe refresh.

Stateless: per-stream bookkeeping (prev frame, prev mask, frame counter)
lives in a state dict the caller threads through, so one detector can
serve many streams — same contract as the model's streaming API.
"""
import torch
import torch.nn.functional as F


class ChangeDetector:
    """Thresholds only; all ops are pixel-domain and negligible next to
    the backbone."""

    def __init__(self, patch_size=16, tau_on=0.02, tau_off=0.01,
                 dilate=True, keyframe_every=30, refresh="keyframe"):
        assert tau_on >= tau_off, "hysteresis requires tau_on >= tau_off"
        assert refresh in ("keyframe", "rolling"), refresh
        self.p = patch_size
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.dilate = dilate
        self.keyframe_every = keyframe_every
        # "keyframe": every patch refreshes on the same frame, every K frames.
        # "rolling": 1/K of the patches refresh on every frame, in a fixed
        # rotation. Both refresh a patch once per K frames and cost the same
        # amortised compute; only the second avoids putting the whole refresh
        # into one frame, which is a discontinuity in the output sequence and
        # shows up as a spike in t-delta, OPW and TCE (Section 5.8).
        self.refresh = refresh

    def is_keyframe(self, st):
        if st is None:
            return True                      # stream start: nothing to diff
        if self.refresh == "rolling":
            return False                     # no frame is globally refreshed
        return st["frame_idx"] % self.keyframe_every == 0

    def rolling_stripe(self, n, frame_idx, device):
        """Patch indices due for refresh on this frame, as a 0/1 row."""
        idx = torch.arange(n, device=device)
        phase = frame_idx % self.keyframe_every
        return (idx % self.keyframe_every == phase).float()

    @torch.no_grad()
    def __call__(self, frame, st=None):
        """frame: (B, C, H, W) in [0, 1]; st: dict from previous call or
        None (stream start). Returns (mask (B, N) float 0/1, st).
        First frame and keyframes are fully active."""
        B, C, H, W = frame.shape
        gh, gw = H // self.p, W // self.p

        if self.is_keyframe(st):
            score = None  # gate will emit all-active; skip the diff
        else:
            diff = (frame - st["prev_frame"]) ** 2  # (B, C, H, W)
            # mean squared diff per patch
            score = F.avg_pool2d(diff.mean(1, keepdim=True), self.p)[:, 0]
        mask, st = self.gate(score, B, gh, gw, frame.device, st)
        st["prev_frame"] = frame.clone()
        return mask, st

    @torch.no_grad()
    def gate(self, score, B, gh, gw, device, st=None):
        """Threshold a per-patch change score (B, gh, gw) into an active
        mask with hysteresis, dilation and keyframe refresh. score=None
        forces a keyframe (no reference yet). The score source is
        pluggable: pixel MSE above, feature-level L1 in GMC mode (§3.5).
        Returns (mask, st)."""
        if score is None or self.is_keyframe(st):
            mask = torch.ones(B, gh * gw, device=device)
        else:
            prev = st["prev_mask"].view(B, gh, gw)
            # hysteresis: active stays active until below tau_off
            m = torch.where(prev > 0.5, (score > self.tau_off), (score > self.tau_on)).float()
            if self.dilate:
                m = F.max_pool2d(m.unsqueeze(1), 3, stride=1, padding=1)[:, 0]
            mask = m.reshape(B, gh * gw)
            if self.refresh == "rolling":
                stripe = self.rolling_stripe(gh * gw,
                                             st["frame_idx"] if st else 0,
                                             device)
                mask = torch.maximum(mask, stripe.unsqueeze(0).expand_as(mask))

        return mask, {"prev_mask": mask.clone(),
                      "frame_idx": (st["frame_idx"] if st else 0) + 1}
