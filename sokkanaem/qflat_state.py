"""Development-only GT-flat retention and bounded refresh-state carry."""
import torch
import torch.nn.functional as F
from sokkanaem.sharpness import norm_disp, grad_mag, boundary_band, _masked_quantile, _dilate
from scripts.qquality_study import MotionStream


@torch.no_grad()
def flat_mask(gt, valid):
    valid = valid.bool() & torch.isfinite(gt) & (gt > 0) & (gt < 150)
    gradient, defined = grad_mag(norm_disp(gt, valid), valid)
    enough = defined.flatten(1).sum(1).reshape(-1, 1, 1, 1) >= 10
    threshold = _masked_quantile(gradient, defined.bool(), .5)
    edge = _dilate(boundary_band(gt, valid, dilate=3), 3)
    mask = (gradient <= threshold) & defined.bool() & enough & ~edge
    return mask, valid


def flat_excess_loss(pred, teacher, valid, mask):
    """Only penalize excess normalized-disparity roughness over Q0 in GT-flat areas."""
    gp, _ = grad_mag(norm_disp(pred, valid), valid)
    gt, _ = grad_mag(norm_disp(teacher.detach(), valid), valid)
    n = mask.flatten(1).sum(1)
    loss = ((gp-gt).relu()*mask).flatten(1).sum(1)/n.clamp_min(1)
    keep = n >= 10
    return (loss*keep).sum()/keep.sum().clamp_min(1)


def refresh_memory(fresh, old, frame, previous, carry, hard_reset=False):
    """Refresh features exactly, retain .9h only at low RGB-change locations.

    No motion alignment/occlusion guarantee: this is a conservative pixel-grid
    heuristic. Never changes fresh Q0 features, pooled calibration or depth.
    """
    if not carry or old is None or hard_reset or old["h"].shape != fresh["h"].shape:
        return fresh, 0
    gh, gw = fresh["grid"]
    size = (gh*14, gw*14)
    a = F.interpolate(frame, size, mode="bilinear", align_corners=False)
    b = F.interpolate(previous, size, mode="bilinear", align_corners=False)
    error = F.avg_pool2d((a-b).square().mean(1, keepdim=True), 14).reshape(-1)
    keep = (error <= .002).to(old["h"].dtype)
    h = old["h"]*.9*keep[:, None, None]
    return {**fresh, "h": h}, int(keep.sum())


class PersistentStream(MotionStream):
    def __init__(self, exact, adapter, carry=False):
        super().__init__(exact, adapter, refresh_every=2)
        self.carry = carry

    @torch.no_grad()
    def step(self, frame, state=None):
        previous = state
        n = 0 if state is None else state["frame_number"]
        pred, state, info = super().step(frame, state)
        retained = 0
        if info["full_refresh"] and previous is not None:
            state["features"], retained = refresh_memory(state["features"], previous["features"], frame,
                                                        previous["previous"], self.carry, hard_reset=n % 32 == 0)
        state["frame_number"] = n+1
        info["retained_h_locations"] = retained
        return pred, state, info
