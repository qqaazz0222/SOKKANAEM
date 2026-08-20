"""Low-Res Global Motion Compensation (IDEA.md §3.5, stage 1).

Estimates camera ego-motion as a homography from a heavily downscaled
frame pair (sparse corners + LK tracking + RANSAC), then warps the
previous full-res frame onto the current view. Pure RGB, no sensors.
Any failure (no texture, degenerate fit) falls back to identity — the
feature gate then simply sees more active patches, never wrong ones.
"""
import numpy as np
import torch
import torch.nn.functional as F


class GlobalMotionCompensator:
    """Weightless. Cost is dominated by the low-res corner/LK pass (~1-2ms)."""

    def __init__(self, lowres=128, max_corners=50, min_matches=8):
        self.lowres = lowres
        self.max_corners = max_corners
        self.min_matches = min_matches
        # how often the estimate degenerates matters: a fallback raises
        # activity rather than suppressing change, so the skip ratio reported
        # for a moving camera is only meaningful next to this rate
        self.calls = 0
        self.fallbacks = 0

    @torch.no_grad()
    def __call__(self, prev, cur):
        """prev, cur: (B, 3, H, W) in [0,1]. Returns prev warped onto cur."""
        import cv2  # optional dep ([video] extra); import here so the
        # fixed-camera path never needs it
        B, _, H, W = prev.shape
        s = self.lowres
        pg = self._gray(prev, s)
        cg = self._gray(cur, s)
        # ponytail: per-sample python loop; B=1 at inference, batch it if
        # training-time GMC ever becomes a thing.
        out = []
        for b in range(B):
            Hm = self._homography(pg[b], cg[b], cv2)
            self.calls += 1
            if Hm is None:
                self.fallbacks += 1
                out.append(prev[b:b + 1])
                continue
            # lift low-res homography to full res: S @ H @ S^-1
            S = np.diag([W / s, H / s, 1.0])
            Hf = S @ Hm @ np.linalg.inv(S)
            out.append(self._warp(prev[b:b + 1], np.linalg.inv(Hf)))
        return torch.cat(out)

    @staticmethod
    def _gray(x, s):
        g = F.interpolate(x, (s, s), mode="bilinear", align_corners=False)
        return (g.mean(1) * 255).byte().cpu().numpy()

    def _homography(self, pg, cg, cv2):
        """prev-gray -> cur-gray homography (low-res pixel coords), or None."""
        pts = cv2.goodFeaturesToTrack(pg, self.max_corners, 0.01, 8)
        if pts is None or len(pts) < self.min_matches:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(pg, cg, pts, None)
        good = st[:, 0] == 1
        if good.sum() < self.min_matches:
            return None
        Hm, _ = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 3.0)
        return Hm

    @staticmethod
    def _warp(img, Hinv):
        """Sample img at Hinv-mapped coords (cur pixel -> prev pixel).
        Out-of-view = zeros -> big feature diff -> patch goes active,
        which is correct: it is genuinely new content."""
        _, _, H, W = img.shape
        dev = img.device
        Hi = torch.as_tensor(Hinv, dtype=torch.float32, device=dev)
        ys, xs = torch.meshgrid(
            torch.arange(H, device=dev, dtype=torch.float32),
            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        pts = torch.stack([xs, ys, torch.ones_like(xs)], -1) @ Hi.T
        w = pts[..., 2:]
        w = torch.where(w.abs() < 1e-6, torch.full_like(w, 1e-6), w)
        src = pts[..., :2] / w
        grid = torch.stack([src[..., 0] / (W - 1) * 2 - 1,
                            src[..., 1] / (H - 1) * 2 - 1], -1)
        return F.grid_sample(img, grid.unsqueeze(0), mode="bilinear",
                             padding_mode="zeros", align_corners=True)
