"""Training-free RGB flow/output-cache controls; NOT an SSM contribution.

Consistency and photometric tests are heuristics, not occlusion certificates.
All CPU flow and transfers occur inside step, hence inside the study timer.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def gray_image(frame, size=128):
    if frame.ndim != 4 or frame.shape[:2] != (1, 3):
        raise ValueError("One RGB frame required")
    small = F.interpolate(frame.float(), size=(size, size), mode="bilinear", align_corners=False)
    rgb = (small[0].permute(1, 2, 0).clamp(0, 1)*255).round().byte().cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def correspondence(previous, current, engine, fb_limit=1.5, photo_limit=12.75):
    """Backward flow in low-resolution pixels; eroded heuristic confidence."""
    backward = engine.calc(current, previous, None)
    forward = engine.calc(previous, current, None)
    h, w = current.shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    mx, my = x + backward[..., 0], y + backward[..., 1]
    inside = (mx >= 0) & (mx <= w-1) & (my >= 0) & (my <= h-1)
    transported_forward = cv2.remap(forward, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fb = np.linalg.norm(backward + transported_forward, axis=-1)
    warped_rgb = cv2.remap(previous, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    photo = np.abs(current.astype(np.float32) - warped_rgb.astype(np.float32))
    reliable = inside & np.isfinite(backward).all(-1) & (fb <= fb_limit) & (photo <= photo_limit)
    reliable = cv2.erode(reliable.astype(np.uint8), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return backward, reliable


def warp_depth(depth, backward, reliable, mode="nearest"):
    """Resize displacement with pixel-unit conversion; sample previous depth."""
    if mode not in ("nearest", "bilinear"):
        raise ValueError("Unsupported depth interpolation")
    h, w = depth.shape[-2:]; fh, fw = backward.shape[:2]
    flow = torch.from_numpy(backward.copy()).to(device=depth.device, dtype=depth.dtype)
    flow = F.interpolate(flow.permute(2, 0, 1)[None], (h, w), mode="bilinear", align_corners=False)
    flow = flow * depth.new_tensor([w/fw, h/fh])[None, :, None, None]
    y, x = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing="ij")
    sx, sy = x + flow[:, 0], y + flow[:, 1]
    finite = torch.isfinite(sx) & torch.isfinite(sy)
    inside = finite & (sx >= 0) & (sx <= w-1) & (sy >= 0) & (sy <= h-1)
    grid = torch.stack((2*(sx+.5)/w-1, 2*(sy+.5)/h-1), -1)
    grid = torch.nan_to_num(grid, nan=2., posinf=2., neginf=-2.)
    warped = F.grid_sample(depth, grid, mode=mode, padding_mode="border", align_corners=False)
    mask = torch.from_numpy(reliable.copy()).to(device=depth.device, dtype=depth.dtype)[None, None]
    mask = F.interpolate(mask, (h, w), mode="nearest").bool() & inside[:, None]
    return warped, mask


class RGBFlowStream(nn.Module):
    """K2 output transport, optionally hold unreliable pixels/full-refresh.

    The RGB/depth cache always describes the immediately preceding frame.
    No hidden history is retained by DIS (each calc receives flow=None).
    """
    def __init__(self, exact, mode="nearest", gated=False, max_unreliable=1., size=128):
        super().__init__()
        if mode not in ("nearest", "bilinear") or not 0 <= max_unreliable <= 1:
            raise ValueError("Invalid policy")
        if size < 64:
            raise ValueError("DIS study requires size >=64")
        self.exact = exact; self.mode = mode; self.gated = gated
        self.max_unreliable = max_unreliable; self.size = size
        self.engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

    @torch.no_grad()
    def step(self, frame, state=None):
        current = gray_image(frame, self.size)
        reset = state is None or state["depth"].shape[-2:] != frame.shape[-2:]
        refresh = reset or state["age"] >= 1
        unreliable = None; reason = "initial" if reset else "periodic"
        if not refresh:
            backward, reliable = correspondence(state["gray"], current, self.engine)
            warped, mask = warp_depth(state["depth"], backward, reliable, self.mode)
            unreliable = 1-float(mask.float().mean())
            refresh = unreliable > self.max_unreliable
            reason = "confidence" if refresh else "reuse"
            if not refresh:
                depth = torch.where(mask, warped, state["depth"]) if self.gated else warped
        if refresh:
            depth, _, _ = self.exact.step(frame)
        next_state = {"depth": depth.detach().clone(), "gray": current.copy(),
                      "age": 0 if refresh else state["age"]+1}
        return depth, next_state, {"full_refresh": refresh, "backbone_calls": int(refresh),
                                  "unreliable_fraction": unreliable, "reason": reason,
                                  "mode": "rgb_flow_output_control_not_ssm"}
