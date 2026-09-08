"""Additional paper-study helpers; frozen v1 model/scoring files stay unchanged."""
from contextlib import contextmanager
import copy
import math

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .alignment import align
from .metrics import clip_scores, warp


LONG_ARMS = {
    "sparse_k30": {"keyframe_every": 30},
    "dense_carry": {"dense": True},
    "dense_reset": {"dense": True, "reset": True},
    "sparse_k5": {"keyframe_every": 5},
    "sparse_k10": {"keyframe_every": 10},
    "sparse_k60": {"keyframe_every": 60},
    "delta_sp": {"forced": True, "spatial_cache": True, "temporal_cache": False},
    "drop_sp": {"forced": True, "spatial_cache": True, "temporal_cache": False,
                "gate_mode": "drop"},
    "delta_no_sp": {"forced": True, "spatial_cache": False, "temporal_cache": False},
    "drop_no_sp": {"forced": True, "spatial_cache": False, "temporal_cache": False,
                   "gate_mode": "drop"},
    "temporal_cache_only": {"forced": True, "spatial_cache": False, "temporal_cache": True},
    "both_cache_replay": {"forced": True, "spatial_cache": True, "temporal_cache": True},
    "output_hold": {"forced": True, "output_hold": True,
                    "spatial_cache": False, "temporal_cache": False},
}
BENCH_ARMS = ("both_cache_replay", "delta_sp", "drop_sp", "delta_no_sp",
              "drop_no_sp", "temporal_cache_only", "output_hold")


class LinearConvCounter:
    """Executed Linear/Conv2d MAC only; does not count fused scan or elementwise ops."""
    def __init__(self, model):
        self.total = 0
        self.handles = []
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                self.handles.append(module.register_forward_hook(self.count))

    def count(self, module, args, output):
        if isinstance(module, torch.nn.Linear):
            self.total += args[0].numel() * module.out_features
        else:
            self.total += output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)

    def close(self):
        for handle in self.handles:
            handle.remove()


def output_hold(current, previous, mask, patch):
    """Patchwise final-output hold, not a sparse implementation of the predictor."""
    gh, gw = current.shape[-2] // patch, current.shape[-1] // patch
    pixel_mask = F.interpolate(mask.reshape(-1, 1, gh, gw),
                               size=current.shape[-2:], mode="nearest").bool()
    return torch.where(pixel_mask, current, previous)


@torch.no_grad()
def native_predict(model, frames, arm, forced=None, count=False):
    device = next(model.parameters()).device
    frames = frames.to(device)
    t, _, height, width = frames.shape
    ones = frames.new_ones(1, t, (height // model.p) * (width // model.p))
    counter = LinearConvCounter(model) if count else None
    try:
        if arm.get("output_hold"):
            if forced is None or not bool(forced[0].all()):
                raise ValueError("output hold needs a dense first frame and a fixed mask")
            previous, out = None, []
            for i in range(t):
                mask = forced[i:i + 1]
                if previous is None or bool(mask.any()):
                    current, _ = model.forward_clip(frames[i:i + 1][None],
                                                     force_mask=ones[:, :1])
                    current = current[:, 0]
                    previous = current if previous is None else output_hold(current, previous, mask, model.p)
                out.append(previous[0])
            pred, masks = torch.stack(out), forced
        elif arm.get("reset"):
            out = [model.forward_clip(frames[i:i + 1][None], force_mask=ones[:, :1])[0][0, 0]
                   for i in range(t)]
            pred, masks = torch.stack(out), ones[0]
        else:
            mask = ones if arm.get("dense") else forced[None] if arm.get("forced") else None
            pred, masks = model.forward_clip(frames[None], force_mask=mask)
            pred, masks = pred[0], masks[0]
        return pred, masks, counter.total if counter else None
    finally:
        if counter:
            counter.close()


def original_views(pairs, scoring_size=256):
    """Source-resolution RGB with the exact ROI induced by the scoring resize/crop.

    The source ROI is expressed in original continuous coordinates. Its output
    grid is at source resolution, so official-input rows do not upscale an
    already reduced 256-pixel RGB image. GT always uses the frozen loader.
    """
    images = []
    for rgb, _ in pairs:
        with Image.open(rgb) as image:
            image = image.convert("RGB")
            w, h = image.size
            scale = scoring_size / min(w, h)
            rw, rh = max(scoring_size, round(w * scale)), max(scoring_size, round(h * scale))
            left, top = (rw - scoring_size) // 2, (rh - scoring_size) // 2
            box = (left * w / rw, top * h / rh,
                   (left + scoring_size) * w / rw, (top + scoring_size) * h / rh)
            n = max(scoring_size, math.ceil(max(box[2] - box[0], box[3] - box[1])))
            images.append(image.resize((n, n), Image.Resampling.BILINEAR, box=box))
    return images


class HFDepth:
    """Pinned local snapshot, per-image inference, no temporal/future-frame inputs."""
    def __init__(self, snapshot, device="cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
        self.common = copy.deepcopy(self.processor)
        self.common.size = {"height": 256, "width": 256}
        self.model, loading = AutoModelForDepthEstimation.from_pretrained(
            snapshot, local_files_only=True, output_loading_info=True)
        self.loading = {key: sorted(loading.get(key, []))
                        for key in ("missing_keys", "unexpected_keys")}
        if loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise RuntimeError(f"baseline checkpoint mismatch: {loading}")
        # DPT's first fusion layer defines a residual branch it never calls.
        # Its four absent parameters are not used by the checkpoint. Guard
        # every missing-parameter module: if any is ever executed, fail the
        # evaluation rather than using an untrained random parameter.
        modules = dict(self.model.named_modules())
        for name in {key.rsplit(".", 1)[0] for key in self.loading["missing_keys"]}:
            if name not in modules:
                raise RuntimeError(f"cannot guard missing parameter module: {name}")
            def reject(module, args, output, name=name):
                raise RuntimeError(f"executed a module with missing checkpoint weights: {name}")
            modules[name].register_forward_hook(reject)
        self.model = self.model.to(device).eval()
        self.metric = type(self.processor).__name__.startswith("ZoeDepth")
        self.space = "depth" if self.metric else "disparity"
        self.device, self.effective = device, {}

    @torch.no_grad()
    def predict(self, frames, originals, official=False):
        proc = self.processor if official else self.common
        images = originals if official else [Image.fromarray((f.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
                                             for f in frames]
        out = []
        # Batching independent images is not temporal context. Batch size 4
        # keeps all three baselines and RAFT on the same 24 GiB GPU safely.
        for i in range(0, len(images), 4):
            batch = images[i:i + 4]
            inp = proc(images=batch, return_tensors="pt").to(self.device)
            self.effective["official" if official else "common"] = list(inp["pixel_values"].shape[-2:])
            post = {"target_sizes": [frames.shape[-2:]] * len(batch)}
            if self.metric:
                post["source_sizes"] = [(im.height, im.width) for im in batch]
            result = proc.post_process_depth_estimation(self.model(**inp), **post)
            out.extend(p["predicted_depth"] for p in result)
        return torch.stack(out).unsqueeze(1)


def grid_from_flow(flow):
    h, w = flow.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=flow.device, dtype=flow.dtype),
                            torch.arange(w, device=flow.device, dtype=flow.dtype), indexing="ij")
    sx, sy = xx + flow[:, 0], yy + flow[:, 1]
    inb = ((sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1))[:, None].float()
    return torch.stack([sx / (w - 1) * 2 - 1, sy / (h - 1) * 2 - 1], -1), inb


def regions_from_flow(gt, valid, flow):
    v = valid.bool()
    ld = torch.log(gt.clamp(min=1e-3))
    gradient = torch.zeros_like(ld)
    for dim in (-1, -2):
        both = v.narrow(dim, 0, gt.shape[dim] - 1) & v.narrow(dim, 1, gt.shape[dim] - 1)
        gradient.narrow(dim, 1, gt.shape[dim] - 1).add_(ld.diff(dim=dim).abs() * both)
    edge = (F.max_pool2d(((gradient > .10) & v).float(), 5, 1, 2) > 0) & v
    median_flow = flow.flatten(2).median(-1).values[..., None, None]
    residual = (flow - median_flow).pow(2).sum(1, keepdim=True).sqrt()
    dynamic = (torch.cat([residual[:1], residual]) > 1.5) & v
    return {"all": v, "edge": edge, "flat": v & ~edge, "dynamic": dynamic,
            "static": v & ~dynamic, "near": v & (gt < 2),
            "mid": v & (gt >= 2) & (gt < 4), "far": v & (gt >= 4)}


def temporal_from_grid(pred, gt, valid, grid, inb):
    """Same equations as metrics.temporal_metrics; one shared RGB flow per clip."""
    mask = inb * valid[1:] * warp(valid[:-1], grid)
    denom = gt[1:].clamp(min=1e-6)
    dp = warp(pred[:-1], grid) - pred[1:]
    dg = warp(gt[:-1], grid) - gt[1:]
    n = mask.sum().clamp(min=1).item()
    opw = ((dp.abs() / denom) * mask).sum().item()
    tce = (((dp - dg).abs() / denom) * mask).sum().item()
    return {"opw": opw / n, "tce": tce / n, "opw_sum": opw, "tce_sum": tce, "warp_px": n}


def frame_sums(depth, gt, valid):
    v = valid.bool()
    g = gt.clamp(min=1e-6)
    dims = (1, 2, 3)
    rel = ((depth - gt).abs() / g * v).sum(dims)
    d1 = ((torch.maximum(depth / g, gt / depth.clamp(min=1e-6)) < 1.25) * v).sum(dims)
    return {"rel_sum": rel.cpu().tolist(), "d1_sum": d1.cpu().tolist(),
            "px": v.sum(dims).cpu().tolist()}


@torch.no_grad()
def score_prediction(frames, pred, gt, valid, space, grid, inb, regions=None,
                     long=False, metric=True):
    from scripts.eval_acc import region_scores
    gauges = (["none"] if metric else []) + ["median", "scaleshift"]
    if long:
        gauges += ["median/frame", "scaleshift/frame"]
    result = {}
    for gauge in gauges:
        per_frame = gauge.endswith("/frame")
        mode = gauge.split("/")[0]
        depth, info = align(pred, gt, valid, mode, space, per_frame)
        sc = clip_scores(frames, depth, gt, valid, temporal=False)
        if not per_frame:
            tm = temporal_from_grid(depth, gt, valid, grid, inb)
            sc.update({k: tm[k] for k in ("opw", "tce")})
            sc["_pooled"].update({k: tm[k] for k in ("opw_sum", "tce_sum", "warp_px")})
        else:
            # Per-frame GT fitting is a shape diagnostic, never a temporal score.
            for key in ("opw", "tce", "temporal_delta"):
                sc[key] = None
        result[gauge] = {"scores": sc, "failed": bool(info["failed"]),
                         "neg_frac": info["neg_frac"], "catastrophic": sc["absrel"] > 1,
                         "regions": region_scores(depth, gt, regions) if regions else None,
                         "frames": frame_sums(depth, gt, valid) if long else None}
    return result
