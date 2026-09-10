"""K4 flow-aligned feature cache with genuine multi-update h recurrence.

Feature-cache history persists in BOTH carry/reset controls. Only the separate
SSM h is ablated. Full Q0 refresh resets both histories every four frames.
"""
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from sokkanaem.detector import ChangeDetector
from sokkanaem.qflow import gray_image, correspondence, warp_depth
from sokkanaem.qflow_features import transport_features


def advance(adapter, state, rgb, mask, flow, reliable, reset_h=False, no_delta=False):
    aligned, _ = transport_features(state, flow, reliable, "nearest")
    if reset_h:
        aligned = {**aligned, "h": torch.zeros_like(aligned["h"])}
    before = float(aligned["h"].detach().abs().mean())
    if no_delta: return aligned, {"updated_patches": 0, "incoming_h_abs_mean": before}
    updated, info = adapter.update(rgb, mask, aligned)
    return updated, {**info, "incoming_h_abs_mean": before}


class RecurrentFlowStream(nn.Module):
    def __init__(self, exact, adapter, reset_h=False, no_delta=False):
        super().__init__()
        self.exact = exact; self.adapter = adapter; self.reset_h = reset_h; self.no_delta = no_delta
        self.engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.detector = ChangeDetector(patch_size=14, tau_on=.001, tau_off=.0005, keyframe_every=4, dilate=True)

    @torch.no_grad()
    def step(self, frame, state=None):
        gray = gray_image(frame)
        image = F.interpolate(frame, (518, 518), mode="bilinear", align_corners=False)
        if state is not None and state["features"]["size"] != frame.shape[-2:]: state = None
        force = state is None or state["age"] >= 3
        score = None if force else F.avg_pool2d((image-state["anchor"]).square().mean(1, keepdim=True), 14)[:,0]
        mask, det = self.detector.gate(score, 1, 37, 37, frame.device, None if state is None else state["det"])
        if force:
            features = self.adapter.refresh(self.exact.encode(frame))
            age = 0; anchor = image.clone(); info = {"updated_patches": 0, "incoming_h_abs_mean": 0.}
        else:
            flow, reliable = correspondence(state["gray"], gray, self.engine)
            features, info = advance(self.adapter, state["features"], frame, mask, flow, reliable, self.reset_h, self.no_delta)
            age = state["age"]+1; anchor = state["anchor"]
        pred = self.exact.decode(features)
        new = {"features": features, "age": age, "anchor": anchor, "det": det, "gray": gray.copy()}
        return pred, new, {**info, "full_refresh": force, "backbone_calls": int(force), "cache_age": age,
                          "active_ratio": float(mask.mean()), "reset_h": self.reset_h, "no_delta": self.no_delta}


class K4OutputFlow(nn.Module):
    """No adapter/decoder on skipped frames; causal output transport control."""
    def __init__(self, exact):
        super().__init__(); self.exact = exact
        self.engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

    @torch.no_grad()
    def step(self, frame, state=None):
        gray = gray_image(frame)
        if state is not None and state["depth"].shape[-2:] != frame.shape[-2:]: state = None
        force = state is None or state["age"] >= 3
        if force:
            pred, _, _ = self.exact.step(frame); age = 0
        else:
            flow, reliable = correspondence(state["gray"], gray, self.engine)
            pred, _ = warp_depth(state["depth"], flow, reliable, "nearest"); age = state["age"]+1
        return pred, {"depth": pred.detach().clone(), "age": age, "gray": gray.copy()}, {
            "full_refresh": force, "backbone_calls": int(force), "cache_age": age}
