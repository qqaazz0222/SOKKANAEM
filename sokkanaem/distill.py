"""Feature distillation from a frozen pretrained DINOv2 encoder.

Zero inference cost — the frozen encoder only runs during training, as an
auxiliary loss target. Rationale: every baseline we compare against (DA v2,
DA3, Video Depth Anything) builds on a DINOv2-family backbone pretrained on
millions of real images; SOKKANAEM's encoder trains from random init on
~195k synthetic clips only. Importing DINOv2's visual representations (not
its depth predictions, not its runtime cost) targets that specific gap.
"""
import torch
import torch.nn.functional as F

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_frozen_dinov2(name="facebook/dinov2-small", device="cpu"):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def dinov2_features(dinov2, frames, grid_hw):
    """frames: (N, 3, H, W) in [0,1]. Returns (N, gh, gw, D), the frozen
    patch features (CLS dropped) resized to grid_hw via bilinear
    interpolation on the feature map — sidesteps DINOv2's 14px patch grid
    vs our 16px one not lining up."""
    dev = frames.device
    x = (frames - _IMAGENET_MEAN.to(dev)) / _IMAGENET_STD.to(dev)
    feat = dinov2(pixel_values=x).last_hidden_state[:, 1:]  # drop CLS
    side = int(round(feat.shape[1] ** 0.5))
    feat = feat.transpose(1, 2).reshape(feat.shape[0], -1, side, side)
    feat = F.interpolate(feat, size=grid_hw, mode="bilinear", align_corners=False)
    return feat.permute(0, 2, 3, 1)  # (N, gh, gw, D)


def load_frozen_teacher(name="depth-anything/Depth-Anything-V2-Small-hf",
                        device="cpu"):
    """Frozen relative-depth teacher. Zero inference cost — it only runs as a
    training target, like the DINOv2 encoder above, but supervises the *output*
    instead of the features. This is the lever for the diagnosed failure mode
    (REPORT §4.17: generalization, not capacity): DA v2 was itself trained on
    62M pseudo-labelled real images, so its disparity map carries real-image
    priors that our ~200k-clip mix does not contain — and it supplies a target
    on every frame, including the real sequences whose Kinect GT is sparse."""
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def teacher_disparity(teacher, frames):
    """frames: (N, 3, H, W) in [0,1]. Returns (N, 1, H, W) relative disparity
    at the input resolution (the teacher head runs at its own scale)."""
    dev = frames.device
    x = (frames - _IMAGENET_MEAN.to(dev)) / _IMAGENET_STD.to(dev)
    d = teacher(pixel_values=x).predicted_depth  # relative disparity, (N,H,W)
    if d.dim() == 3:
        d = d.unsqueeze(1)
    return F.interpolate(d, size=frames.shape[-2:], mode="bilinear",
                         align_corners=False)


def affine_invariant_loss(pred_depth, teacher_disp, valid=None):
    """Scale-and-shift invariant L1 between our disparity and the teacher's.

    The teacher predicts *relative* disparity with an arbitrary affine gauge,
    so anything that compares raw values is meaningless. Both sides are
    normalized MiDaS style (median centre, mean-absolute-deviation scale) and
    then matched — this trains geometry/ordering without letting the teacher
    overwrite the metric scale our GT supervision provides."""
    def norm(x, v):
        v = v.bool() if v is not None else torch.ones_like(x, dtype=torch.bool)
        if not bool(v.any()):
            return x * 0
        t = x[v].median()
        s = (x[v] - t).abs().mean().clamp(min=1e-6)
        return (x - t) / s

    p = norm(1.0 / pred_depth.clamp(min=1e-3), valid)
    t = norm(teacher_disp, valid)
    if valid is None:
        return (p - t).abs().mean()
    return ((p - t).abs() * valid).sum() / valid.sum().clamp(min=1)


def distill_loss(tokens, proj, target_feat):
    """tokens: (B, N, dim) our encoder tokens (any block, pre-decoder).
    proj: Linear(dim -> D), trainable. target_feat: (B, gh, gw, D) frozen,
    gh*gw == N. Cosine loss — scale-free, since dim << D so matching
    magnitude isn't meaningful, only direction/relative structure."""
    B, N, _ = tokens.shape
    pred = proj(tokens)
    tgt = target_feat.reshape(B, N, -1)
    return (1 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
