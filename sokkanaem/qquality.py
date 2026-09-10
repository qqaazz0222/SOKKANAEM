"""Development-only Q0 output retention and inexpensive local motion alignment.

No edits to the frozen Q0 or native paper model. Motion is a local RGB matcher,
not optical-flow ground truth or a calibrated occlusion estimator.
"""
import torch
from torch.nn import functional as F


def decode_with_grad(exact, encoded):
    """Frozen decoder weights, but gradients to predicted features/pooled state."""
    disp = exact.q._decode(encoded["features"], encoded["grid"])
    return exact.q._calibrate(disp, encoded["pooled"], encoded["size"])


def depth_retention(pred, teacher):
    """Metric log-depth and multiscale log-gradient retention; no GT fitting."""
    p, t = pred.clamp_min(1e-4).log(), teacher.detach().clamp_min(1e-4).log()
    depth = (p - t).abs().mean()
    terms = []
    for stride in (1, 2, 4):
        a, b = p[..., ::stride, ::stride], t[..., ::stride, ::stride]
        for dim in (-1, -2):
            da, db = torch.diff(a, dim=dim), torch.diff(b, dim=dim)
            weight = 1 + 4 * db.abs().clamp(max=.25)
            terms.append((weight * (da - db).abs()).mean())
    edge = torch.stack(terms).mean()
    return depth, edge


@torch.no_grad()
def motion_grid(previous, current, target_grid, radius=2, match_size=74):
    """Backward local block match in 74px RGB; displacement radius = 14px at518.

    Matching minimizes 3x3-window RGB MSE with a small zero-motion preference.
    Returns grid_sample coordinates and photometric error at the feature grid.
    Global scene changes/large motion cannot be repaired by this matcher.
    """
    if previous.shape != current.shape or previous.shape[0] != 1:
        raise ValueError("One same-sized previous/current frame is required")
    if radius < 0 or match_size < 3:
        raise ValueError("Invalid motion settings")
    a = F.interpolate(previous.float(), (match_size, match_size), mode="bilinear", align_corners=False)
    b = F.interpolate(current.float(), (match_size, match_size), mode="bilinear", align_corners=False)
    padded = F.pad(a, (radius, radius, radius, radius), mode="replicate")
    offsets = [(0, 0)] + [(dy, dx) for dy in range(-radius, radius+1)
                          for dx in range(-radius, radius+1) if dy or dx]
    costs = []
    yy, xx = torch.meshgrid(torch.arange(match_size, device=a.device),
                            torch.arange(match_size, device=a.device), indexing="ij")
    for dy, dx in offsets:
        shifted = padded[..., radius+dy:radius+dy+match_size, radius+dx:radius+dx+match_size]
        cost = F.avg_pool2d((shifted-b).square().mean(1, keepdim=True), 3, 1, 1)
        valid = (yy+dy >= 0) & (yy+dy < match_size) & (xx+dx >= 0) & (xx+dx < match_size)
        costs.append(cost + 1e-4*(dx*dx+dy*dy) + (~valid)[None, None]*1e3)
    cost, index = torch.cat(costs, 1).min(1, keepdim=True)
    offsets = a.new_tensor(offsets)
    flow = offsets[index[:, 0]].permute(0, 3, 1, 2)
    flow = F.interpolate(flow, target_grid, mode="bilinear", align_corners=False)
    gh, gw = target_grid
    y, x = torch.meshgrid((torch.arange(gh, device=a.device)+.5)*2/gh-1,
                          (torch.arange(gw, device=a.device)+.5)*2/gw-1, indexing="ij")
    grid = torch.stack((x, y), -1)[None] + flow[:, [1, 0]].permute(0, 2, 3, 1)*2/match_size
    return grid, F.interpolate(cost, target_grid, mode="bilinear", align_corners=False)


def align_state(state, grid):
    """Transport patch caches AND h. CLS/pooled are global and left unchanged.

    Inactive update-copy semantics refer to this aligned state, not old pixels.
    Bilinear transport of SSM memory is an approximation requiring ablation.
    """
    gh, gw = state["grid"]
    if grid.shape != (1, gh, gw, 2):
        raise ValueError("Motion grid does not match the feature cache")
    def warp(x):
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    features = []
    for f in state["features"]:
        patches = f[:, 1:].transpose(1, 2).reshape(1, -1, gh, gw)
        patches = warp(patches).flatten(2).transpose(1, 2)
        features.append(torch.cat((f[:, :1], patches), 1))
    h = state["h"]
    transported = warp(h.reshape(gh, gw, -1).permute(2, 0, 1)[None])
    transported = transported[0].permute(1, 2, 0).reshape_as(h)
    return {**state, "features": features, "h": transported}
