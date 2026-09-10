"""Apply the same RGB DIS correspondence to Q0 features and SSM memory.

This is an untrained architectural transfer of an existing adapter; it is not
evidence that an RGB displacement exactly transports contextual ViT features.
"""
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from sokkanaem.detector import ChangeDetector
from sokkanaem.qflow import gray_image, correspondence, warp_depth


def transport_features(state, flow, reliable, mode="nearest"):
    gh, gw = state["grid"]
    fs = state["features"]; h = state["h"]
    if any(f.shape[0] != 1 or f.shape[1] != gh*gw+1 for f in fs):
        raise ValueError("One stream with CLS plus grid patches required")
    planes = [f[:, 1:].transpose(1, 2).reshape(1, -1, gh, gw) for f in fs]
    planes.append(h.reshape(gh, gw, -1).permute(2, 0, 1)[None])
    joined = torch.cat(planes, 1)
    transported, _ = warp_depth(joined, flow, reliable, mode)
    split = transported.split([p.shape[1] for p in planes], dim=1)
    features = [torch.cat((f[:, :1], p.flatten(2).transpose(1, 2)), 1) for f, p in zip(fs, split[:-1])]
    hidden = split[-1][0].permute(1, 2, 0).reshape_as(h)
    changed = (split[0] != planes[0]).any(1).float().mean()
    return {**state, "features": features, "h": hidden}, float(changed)


class RGBFlowFeatureStream(nn.Module):
    def __init__(self, exact, adapter, mode="nearest", no_delta=False):
        super().__init__()
        if mode not in ("nearest", "bilinear"):
            raise ValueError("Unsupported interpolation")
        self.exact = exact; self.adapter = adapter; self.mode = mode; self.no_delta = no_delta
        self.engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.detector = ChangeDetector(patch_size=14, tau_on=.001, tau_off=.0005, keyframe_every=2, dilate=True)

    @torch.no_grad()
    def step(self, frame, state=None):
        if frame.ndim != 4 or frame.shape[:2] != (1, 3):
            raise ValueError("One RGB stream required")
        image = F.interpolate(frame, (518, 518), mode="bilinear", align_corners=False)
        gray = gray_image(frame)
        if state is not None and state["features"]["size"] != frame.shape[-2:]:
            state = None
        force = state is None or state["age"] >= 1
        score = None if force else F.avg_pool2d((image-state["anchor"]).square().mean(1, keepdim=True), 14)[:, 0]
        mask, det = self.detector.gate(score, 1, 37, 37, frame.device, None if state is None else state["det"])
        updated = 0; moved = None
        if force:
            features = self.adapter.refresh(self.exact.encode(frame))
            anchor = image.clone(); age = 0
        else:
            flow, reliable = correspondence(state["gray"], gray, self.engine)
            features, moved = transport_features(state["features"], flow, reliable, self.mode)
            if not self.no_delta:
                features, info = self.adapter.update(frame, mask, features)
                updated = info["updated_patches"]
            anchor = state["anchor"]; age = state["age"]+1
        pred = self.exact.decode(features)
        next_state = {"features": features, "anchor": anchor, "age": age, "det": det, "gray": gray.copy()}
        return pred, next_state, {"full_refresh": force, "backbone_calls": int(force), "active_ratio": float(mask.mean()),
                                  "updated_patches": updated, "transport_changed_fraction": moved,
                                  "mode": self.mode, "no_delta": self.no_delta, "cache_age": age}
