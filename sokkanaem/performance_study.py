"""Task 9–11 adapters: real streaming paths, independent of frozen v1 code."""
from collections import defaultdict
from contextlib import contextmanager
import copy
import hashlib
import itertools
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from . import from_checkpoint
from .detector import ChangeDetector
from .study import HFDepth, output_hold

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "work_dirs/paper_dependencies"
BASE = ROOT / "work_dirs/v11-longclip-spread-s0/latest.pt"
DA2 = Path("/home/hyunsu/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small-hf/snapshots/5426e4f0f36572d16453bbda7a8389317b1bef99")
POINTS = {
    **{f"sparse_k{k}": {"kind": "native", "keyframe_every": k} for k in (5, 10, 30, 60)},
    "dense_carry": {"kind": "native", "dense": True},
    "dense_reset": {"kind": "native", "dense": True, "reset": True},
    "output_hold": {"kind": "native", "hold": True},
    "gmc_k30": {"kind": "native", "gmc": True, "tau_on": .1, "tau_off": .05},
    **{f"da2_{size}": {"kind": "da2", "size": size, "original": size != 256}
       for size in (196, 256, 350, 518)},
    **{f"midas_{size}": {"kind": "midas", "size": size, "original": size != 256}
       for size in (192, 256, 384)},
}


def rgb_view(image, original=False):
    """Same continuous ROI as study.original_views, or exact frozen 256 crop."""
    w, h = image.size
    scale = 256 / min(w, h)
    rw, rh = max(256, round(w * scale)), max(256, round(h * scale))
    left, top = (rw - 256) // 2, (rh - 256) // 2
    if not original:
        return image.resize((rw, rh), Image.Resampling.BILINEAR).crop((left, top, left + 256, top + 256))
    box = (left * w / rw, top * h / rh, (left + 256) * w / rw, (top + 256) * h / rh)
    n = max(256, int(np.ceil(max(box[2] - box[0], box[3] - box[1]))))
    return image.resize((n, n), Image.Resampling.BILINEAR, box=box)


def persistent_bytes(value):
    """Unique tensor storages, recursively; detector state and aliases matter."""
    seen, containers = set(), set()
    def visit(obj):
        if torch.is_tensor(obj):
            storage = obj.untyped_storage()
            key = (str(obj.device), storage.data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(obj, (dict, tuple, list)):
            if id(obj) in containers:
                return 0
            containers.add(id(obj))
            return sum(visit(v) for v in (obj.values() if isinstance(obj, dict) else obj))
        return 0
    return visit(value)


class StageTimer:
    """Diagnostic synchronization at component boundaries, NOT primary timing."""
    def __init__(self, device="cuda"):
        self.device = device
        self.values = defaultdict(float)

    def sync(self):
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def call(self, name, fn, *args, **kwargs):
        self.sync()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        self.sync()
        self.values[name] += (time.perf_counter() - start) * 1000
        return result


class TimedObject:
    def __init__(self, wrapped, timer, name):
        self.wrapped, self.timer, self.name = wrapped, timer, name

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __call__(self, *args, **kwargs):
        return self.timer.call(self.name, self.wrapped, *args, **kwargs)

    def gate(self, *args, **kwargs):
        return self.timer.call(self.name, self.wrapped.gate, *args, **kwargs)


class AllActive:
    """Dense deployment has no reason to compute a pixel change detector."""
    def __init__(self, p):
        self.p = p

    def __call__(self, frame, state=None):
        b, _, h, w = frame.shape
        return frame.new_ones(b, (h // self.p) * (w // self.p)), None

    def is_keyframe(self, state):
        return True


class Native:
    space, metric, original, size = "depth", True, False, 256

    def __init__(self, spec, checkpoint=BASE, device="cuda", timer=None):
        self.spec, self.device, self.timer = spec, device, timer
        kwargs = {k: v for k, v in spec.items() if k in
                  ("keyframe_every", "gmc", "tau_on", "tau_off")}
        if spec.get("dense") or spec.get("hold"):
            kwargs.update(spatial_cache=False, temporal_cache=False)
        self.model = from_checkpoint(checkpoint, device, **kwargs).eval()
        self.detector = None
        if spec.get("hold"):
            self.detector = self.model.detector
        if spec.get("dense") or spec.get("hold"):
            self.model.detector = AllActive(self.model.p)
        if timer:
            self.model.detector = TimedObject(self.model.detector, timer, "detector")
            if self.detector is not None:
                self.detector = TimedObject(self.detector, timer, "detector")
            if self.model.gmc is not None:
                self.model.gmc = TimedObject(self.model.gmc, timer, "gmc")
            for attr, name in (("_forward_tokens", "backbone"), ("_decode", "decoder")):
                method = getattr(self.model, attr)
                setattr(self.model, attr, lambda *a, _fn=method, _name=name, **kw:
                        timer.call(_name, _fn, *a, **kw))
            method = self.model.embed.forward
            self.model.embed.forward = lambda *a, **kw: timer.call("embedding", method, *a, **kw)
        self.params = sum(p.numel() for p in self.model.parameters())
        self.reset()

    def reset(self):
        self.state = None
        import cv2
        cv2.setRNGSeed(0)
        if self.model.gmc is not None:
            gmc = self.model.gmc.wrapped if isinstance(self.model.gmc, TimedObject) else self.model.gmc
            gmc.calls = gmc.fallbacks = 0

    def preprocess(self, image):
        array = np.asarray(rgb_view(image)).copy()
        return torch.from_numpy(array).permute(2, 0, 1).float().div(255).unsqueeze(0)

    @torch.no_grad()
    def infer(self, x):
        if self.spec.get("hold"):
            st = self.state or {"det": None, "output": None}
            keyframe = self.detector.is_keyframe(st["det"])
            mask, det = self.detector(x, st["det"])
            if 0 < self.model.dense_above and mask.mean() > self.model.dense_above:
                mask = torch.ones_like(mask)
            previous = st["output"]
            if previous is None or bool(mask.any()):
                current, _, _ = self.model.step(x, None)
                previous = current if previous is None else output_hold(current, previous, mask, self.model.p)
            self.state = {"det": det, "output": previous}
            return previous, {"keyframe": keyframe, "mask": mask}
        state = None if self.spec.get("reset") else self.state
        keyframe = self.model.detector.is_keyframe(state["det"] if state else None)
        prediction, self.state, info = self.model.step(x, state)
        if self.spec.get("reset"):
            self.state = None  # memoryless deployment need not retain transient SSM state
        return prediction, {**info, "keyframe": keyframe}

    def diagnostics(self, info):
        mask = info["mask"]
        active = mask.mean().item()
        det = self.state.get("det") if self.state else None
        raw = det["prev_mask"].mean().item() if isinstance(det, dict) else active
        gmc = self.model.gmc
        return {"active": active, "keyframe": bool(info["keyframe"]),
                "dense_fallback": active == 1 and raw < 1,
                "state_bytes": persistent_bytes(self.state),
                "gmc_calls": gmc.calls if gmc else 0, "gmc_fallbacks": gmc.fallbacks if gmc else 0}


class Relative:
    space, metric = "disparity", False

    def __init__(self, spec, device="cuda", timer=None):
        self.spec, self.device, self.timer = spec, device, timer
        self.size, self.original, self.state = spec["size"], spec["original"], None
        if spec["kind"] == "da2":
            hf = HFDepth(str(DA2), device)
            self.model = hf.model
            self.processor = copy.deepcopy(hf.processor)
            self.processor.size = {"height": self.size, "width": self.size}
        else:
            for path in (DEPS / "MiDaS-v2_1", DEPS / "gen-efficientnet-pytorch"):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            import geffnet
            from midas import blocks
            from midas.midas_net_custom import MidasNet_small
            from midas.transforms import Resize, NormalizeImage, PrepareForNet
            import cv2
            def encoder(unused, exportable=False):
                with geffnet.config.set_exportable(exportable):
                    model = geffnet.create_model("tf_efficientnet_lite3", pretrained=False)
                return blocks._make_efficientnet_backbone(model)
            # Pin the transitive backbone locally; no torch.hub network execution.
            with patch.object(blocks, "_make_pretrained_efficientnet_lite3", encoder):
                self.model = MidasNet_small(None, features=64, backbone="efficientnet_lite3",
                    exportable=True, non_negative=True, blocks={"expand": True})
            weights = torch.load(DEPS / "midas_v21_small_256.pt", map_location="cpu", weights_only=True)
            self.model.load_state_dict(weights, strict=True)
            self.model = self.model.to(device).eval()
            self.transforms = [Resize(self.size, self.size, resize_target=False, keep_aspect_ratio=True,
                ensure_multiple_of=32, resize_method="upper_bound", image_interpolation_method=cv2.INTER_CUBIC),
                NormalizeImage(mean=[.485, .456, .406], std=[.229, .224, .225]), PrepareForNet()]
        self.params = sum(p.numel() for p in self.model.parameters())

    def reset(self):
        self.state = None

    def preprocess(self, image):
        view = rgb_view(image, self.original)
        if self.spec["kind"] == "da2":
            return self.processor(images=view, return_tensors="pt")["pixel_values"]
        sample = {"image": np.asarray(view) / 255.}
        for transform in self.transforms:
            sample = transform(sample)
        return torch.from_numpy(sample["image"]).unsqueeze(0)

    @torch.no_grad()
    def infer(self, x):
        def run():
            if self.spec["kind"] == "da2":
                out = self.model(pixel_values=x)
                return self.processor.post_process_depth_estimation(out, target_sizes=[(256, 256)])[0]["predicted_depth"][None, None]
            prediction = self.model(x)[:, None]
            return F.interpolate(prediction, (256, 256), mode="bicubic", align_corners=False)
        result = self.timer.call("network_and_output_resize", run) if self.timer else run()
        return result, {}

    def diagnostics(self, info):
        return {"active": None, "keyframe": False, "dense_fallback": False,
                "state_bytes": 0, "gmc_calls": 0, "gmc_fallbacks": 0}


def adapter(spec, checkpoint=BASE, device="cuda", timer=None):
    return Native(spec, checkpoint, device, timer) if spec["kind"] == "native" else Relative(spec, device, timer)


@torch.no_grad()
def run_clip(model, pairs, *, timing=False, profile=False, collect=True):
    """Batch-one PNG path -> CPU 256 depth. Telemetry/checksum are untimed."""
    model.reset()
    frames, durations, telemetry, sizes = [], [], [], []
    checksum = hashlib.sha256()
    timer = model.timer if profile else None
    for rgb, _ in pairs:
        if timer:
            timer.values.clear()
        if timing or timer:
            torch.cuda.synchronize()
        start = time.perf_counter()
        def preprocess():
            with Image.open(rgb) as image:
                return model.preprocess(image.convert("RGB"))
        x = timer.call("read_decode_preprocess", preprocess) if timer else preprocess()
        sizes.append(list(x.shape[-2:]))
        x = timer.call("h2d", x.to, model.device) if timer else x.to(model.device)
        pred, info = model.infer(x)
        cpu = timer.call("d2h", pred.cpu) if timer else pred.cpu()
        # cpu() is blocking; no GPU work belonging to this output remains.
        end = time.perf_counter()
        if not bool(torch.isfinite(cpu).all()):
            raise ValueError("non-finite model output: do not silently exclude a frame")
        checksum.update(cpu.contiguous().numpy().tobytes())
        if collect:
            frames.append(cpu[0])
        durations.append((end - start) * 1000)
        telemetry.append({**model.diagnostics(info), "components_ms": dict(timer.values) if timer else None})
        del x, pred, cpu
    if len({tuple(s) for s in sizes}) != 1:
        raise ValueError("effective input shape changed within a clip")
    return {"prediction": torch.stack(frames) if collect else None,
            "prediction_sha256": checksum.hexdigest(), "ms": durations,
            "telemetry": telemetry, "input_size": sizes[0]}


def paired_sequence_summary(differences):
    """Equal-sequence difference; exact enumeration of small-n bootstrap/sign flips."""
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("paired inference requires at least two finite sequence differences")
    if len(d) > 7:
        raise ValueError("enumeration is intended for this small development sequence set")
    draws = np.asarray(list(itertools.product(range(len(d)), repeat=len(d))))
    bootstrap = d[draws].mean(1)
    flips = np.asarray(list(itertools.product((-1, 1), repeat=len(d))))
    permuted = (flips * d).mean(1)
    lo, hi = np.quantile(bootstrap, [.025, .975])
    p = np.mean(np.abs(permuted) >= abs(d.mean()) - 1e-12)
    return {"difference": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "sign_flip_p": float(p), "sequence_n": len(d), "bootstrap_draws": len(draws),
            "sign_flip_draws": len(flips), "all_sequence_differences": d.tolist()}


def pareto_flags(points):
    """Two-axis descriptive frontier only; failure/temporal metrics remain visible."""
    return [not any((q[0] <= p[0] and q[1] <= p[1]) and q != p for q in points) for p in points]
