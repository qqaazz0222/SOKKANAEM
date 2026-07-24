"""Training losses (IDEA.md §3.3), all masked by depth validity."""
import torch
import torch.nn.functional as F


def si_log_loss(pred, gt, valid):
    d = (torch.log(pred.clamp(min=1e-6)) - torch.log(gt.clamp(min=1e-6))) * valid
    n = valid.sum().clamp(min=1)
    return (d ** 2).sum() / n - 0.5 * (d.sum() / n) ** 2


def grad_loss(pred, gt, valid):
    def gy(x):
        return x[..., 1:, :] - x[..., :-1, :]

    def gx(x):
        return x[..., :, 1:] - x[..., :, :-1]

    vy = valid[..., 1:, :] * valid[..., :-1, :]
    vx = valid[..., :, 1:] * valid[..., :, :-1]
    return (((gy(pred) - gy(gt)).abs() * vy).sum() / vy.sum().clamp(min=1)
            + ((gx(pred) - gx(gt)).abs() * vx).sum() / vx.sum().clamp(min=1))


def normal_loss(pred, gt, valid):
    """1 - cosine similarity between depth-gradient pseudo surface normals.
    No camera intrinsics are tracked anywhere in this codebase (depth is
    just per-pixel meters), so normals live in image-plane coordinates:
    n ∝ (-dz/dx, -dz/dy, 1) rather than true unprojected 3D normals —
    still penalizes cross-object depth-slope mismatches at edges."""
    def normals(d):
        dzdx = (d[..., :, 1:] - d[..., :, :-1])[..., :-1, :]
        dzdy = (d[..., 1:, :] - d[..., :-1, :])[..., :, :-1]
        n = torch.stack([-dzdx, -dzdy, torch.ones_like(dzdx)], dim=-1)
        return F.normalize(n, dim=-1, eps=1e-6)

    # each normal at (i,j) consumes depth at (i,j), (i,j+1), (i+1,j) — all
    # three must be valid or the slope is garbage (sky sentinels in vkitti2/
    # tartanair). Match grad_loss's both-endpoints masking rigor.
    v = valid[..., :-1, :-1] * valid[..., :-1, 1:] * valid[..., 1:, :-1]
    cos = (normals(pred) * normals(gt)).sum(-1)
    return ((1 - cos) * v).sum() / v.sum().clamp(min=1)


def temporal_loss(depths, masks, patch=16):
    """Static regions must keep identical depth across frames."""
    B, T = depths.shape[:2]
    if T < 2:
        return depths.sum() * 0
    static = 1 - masks[:, 1:]                                # (B, T-1, N)
    gh = depths.shape[-2] // patch
    m = static.reshape(B, T - 1, gh, -1)
    m = F.interpolate(m.reshape(B * (T - 1), 1, gh, -1).float(),
                      scale_factor=patch, mode="nearest")
    diff = (depths[:, 1:] - depths[:, :-1]).reshape(B * (T - 1), 1, *depths.shape[-2:])
    return (diff.abs() * m).sum() / m.sum().clamp(min=1)
