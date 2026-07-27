"""Multi-dataset video-depth loading.

Canonical sample (what train.py sees, dataset-agnostic):
    frames (T, 3, H, W) float [0,1]
    depth  (T, 1, H, W) float, meters (0 = invalid)
    valid  (T, 1, H, W) float 0/1

Every dataset reduces to two things:
    1. sequences: list of sequences, each a list of (rgb_path, depth_path)
    2. depth_scale: raw 16-bit PNG value -> meters divisor
Adapters below provide only that; ClipDataset does the rest (clip slicing,
resize, validity mask). New dataset = one ~5-line adapter + registry entry.

Spec strings (CLI): "name:/path" or "folder:/path:scale"
    scannet:/data/scannet     scene*/color/*.jpg + scene*/depth/*.png (mm)
    tum:/data/tum             seq*/rgb/*.png + seq*/depth/*.png (scale 5000)
    bonn:/data/bonn           same layout as tum
    kitti:/data/kitti         drive*/image_02/data/*.png +
                              drive*/proj_depth/groundtruth/image_02/*.png (scale 256)
    vkitti2:/data/vkitti2     Scene*/<variation>/frames/rgb/Camera_*/*.jpg +
                              frames/depth/Camera_*/*.png (cm, scale 100;
                              rgb+depth tars extracted into one root)
    tartanair2:/data/tartanair_v2   <Env>/Data_easy/P*/image_lcam_front/*.png +
                              depth_lcam_front/*.png (depth: RGBA bytes are a
                              packed float32 meters value, not u16 — see
                              "packed_f32" mode below; ~8248 sentinel = sky)
    pointodyssey:/data/pointodyssey  {train,val,test,sample}/<scene>/rgbs/rgb_%05d.jpg +
                              depths/depth_%05d.png (u16, scale 65.535 per the
                              official PIPs++ loader: depth_m = raw/65535*1000)
    folder:/data/mine:1000    generic: */rgb/* + */depth/*, custom scale

Metric ranges differ across datasets (indoor ~10m vs KITTI ~80m); mixing is
safe because training uses scale-invariant log loss.
"""
import bisect
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler


def _pair_sorted(rgb_dir, depth_dir):
    """Pair frames by sorted order (index-aligned naming, the common case)."""
    rgbs = sorted(glob.glob(os.path.join(rgb_dir, "*")))
    deps = sorted(glob.glob(os.path.join(depth_dir, "*")))
    n = min(len(rgbs), len(deps))
    return list(zip(rgbs[:n], deps[:n]))


def _pair_by_name(rgb_dir, depth_dir):
    """Pair by matching basename stem (KITTI: sparse GT misses some frames)."""
    deps = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(depth_dir, "*"))}
    pairs = []
    for r in sorted(glob.glob(os.path.join(rgb_dir, "*"))):
        stem = os.path.splitext(os.path.basename(r))[0]
        if stem in deps:
            pairs.append((r, deps[stem]))
    return pairs


def _pair_by_timestamp(rgb_dir, depth_dir, max_dt=0.02):
    """Pair by nearest capture time (TUM/Bonn: filenames ARE timestamps, the
    two streams have different rates AND different counts, so sorted-order
    pairing silently drifts the GT out of alignment). 0.02s window is TUM's
    own associate.py default; unmatched rgb frames are dropped."""
    def stamped(d):
        out = []
        for p in glob.glob(os.path.join(d, "*")):
            try:
                out.append((float(os.path.splitext(os.path.basename(p))[0]), p))
            except ValueError:
                continue  # not a timestamp-named file
        return sorted(out)

    rgbs, deps = stamped(rgb_dir), stamped(depth_dir)
    if not deps:
        return []
    times = [t for t, _ in deps]
    pairs = []
    for t, r in rgbs:
        j = bisect.bisect_left(times, t)
        k = min((k for k in (j - 1, j) if 0 <= k < len(times)),
                key=lambda k: abs(times[k] - t), default=None)
        if k is not None and abs(times[k] - t) <= max_dt:
            pairs.append((r, deps[k][1]))
    return pairs


def _subdirs_seqs(root, rgb_sub, depth_sub, pair=_pair_sorted):
    seqs = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        rgb, dep = os.path.join(d, rgb_sub), os.path.join(d, depth_sub)
        if os.path.isdir(rgb) and os.path.isdir(dep):
            seqs.append(pair(rgb, dep))
    return seqs


def scannet(root):
    return _subdirs_seqs(root, "color", "depth")


def tum(root):
    """TUM RGB-D layout (Bonn Dynamic and the fr3 *_static sequences share
    it). Timestamp-associated, not sorted-order — see _pair_by_timestamp."""
    return _subdirs_seqs(root, "rgb", "depth", pair=_pair_by_timestamp)


def kitti(root):
    return _subdirs_seqs(root, "image_02/data",
                         "proj_depth/groundtruth/image_02", pair=_pair_by_name)


def vkitti2(root):
    """One sequence per Scene*/variation/Camera_* (clone, fog, ... are
    separate sequences — same road, different appearance)."""
    seqs = []
    for rgb in sorted(glob.glob(os.path.join(root, "*/*/frames/rgb/Camera_*"))):
        frames_dir = os.path.dirname(os.path.dirname(rgb))
        dep = os.path.join(frames_dir, "depth", os.path.basename(rgb))
        if os.path.isdir(dep):
            seqs.append(_pair_sorted(rgb, dep))
    return seqs


def tartanair2(root):
    """One sequence per <Env>/Data_easy/P<NNN> trajectory."""
    seqs = []
    for rgb in sorted(glob.glob(os.path.join(root, "*/Data_easy/P*/image_lcam_front"))):
        dep = os.path.join(os.path.dirname(rgb), "depth_lcam_front")
        if os.path.isdir(dep):
            seqs.append(_pair_sorted(rgb, dep))
    return seqs


def pointodyssey(root):
    """One sequence per {train,val,test,sample}/<scene>."""
    seqs = []
    for rgb in sorted(glob.glob(os.path.join(root, "*/*/rgbs"))):
        dep = os.path.join(os.path.dirname(rgb), "depths")
        if os.path.isdir(dep):
            seqs.append(_pair_sorted(rgb, dep))
    return seqs


# (loader, scale, depth_mode) — depth_mode "u16" (default): 16-bit PNG / scale
# = meters. "packed_f32": RGBA bytes reinterpreted as one packed float32
# meters value (TartanAir V2's format); scale is unused for that mode.
ADAPTERS = {
    "scannet": (scannet, 1000.0, "u16"),
    "tum": (tum, 5000.0, "u16"),
    "bonn": (tum, 5000.0, "u16"),   # identical layout
    "nyu": (tum, 1000.0, "u16"),    # extracted-frames layout
    "kitti": (kitti, 256.0, "u16"),
    "vkitti2": (vkitti2, 100.0, "u16"),  # depth in cm
    "tartanair2": (tartanair2, 1.0, "packed_f32"),
    "pointodyssey": (pointodyssey, 65.535, "u16"),  # raw/65535*1000 = meters
    "folder": (tum, 1000.0, "u16"),  # generic */rgb + */depth, scale via spec
}


class ClipDataset(Dataset):
    """Slices sequences into fixed-length clips; loads and resizes on access.

    augment=True adds clip-consistent geometric + photometric jitter. Clip
    consistent is the whole point: one transform is drawn per clip and applied
    to every frame in it, so temporal structure (and therefore the change
    detector's masks and the temporal loss) stays valid. Without it the model
    fits its training scenes and does not transfer — measured on v7: AbsRel
    0.192 / d1 0.707 on seen clips vs 0.356 / 0.519 on the holdout, with no
    augmentation of any kind in the pipeline.
    """

    def __init__(self, sequences, depth_scale, clip_len=4, frame_stride=1,
                 clip_stride=2, size=128, depth_mode="u16", augment=False):
        self.scale = depth_scale
        self.mode = depth_mode
        self.T = clip_len
        self.size = size
        self.augment = augment
        span = clip_len * frame_stride
        self.clips = [(seq, s, frame_stride)
                      for seq in sequences if len(seq) >= span
                      for s in range(0, len(seq) - span + 1, clip_stride)]

    def __len__(self):
        return len(self.clips)

    def _draw_aug(self):
        """One geometric+photometric draw, reused for every frame of the clip."""
        if not self.augment:
            return None
        return {
            "zoom": random.uniform(0.55, 1.0),  # random-resized-crop scale
            "cx": random.random(), "cy": random.random(),
            "flip": random.random() < 0.5,
            # photometric: RGB only, depth is untouched
            "bright": random.uniform(0.75, 1.3),
            "contrast": random.uniform(0.75, 1.3),
            "sat": random.uniform(0.7, 1.4),
            "gamma": random.uniform(0.8, 1.25),
        }

    def _fit(self, img, resample, aug=None):
        """Aspect-preserving: resize shorter side to self.size, center-crop.
        With aug: crop a random `zoom` fraction at a random position first, so
        the model sees varied scale and framing instead of one fixed view."""
        w, h = img.size
        if aug is not None:
            side = max(8, int(round(min(w, h) * aug["zoom"])))
            left = int(round((w - side) * aug["cx"]))
            top = int(round((h - side) * aug["cy"]))
            img = img.crop((left, top, left + side, top + side))
            w, h = img.size
        s = self.size / min(w, h)
        img = img.resize((max(self.size, round(w * s)),
                          max(self.size, round(h * s))), resample)
        left = (img.width - self.size) // 2
        top = (img.height - self.size) // 2
        img = img.crop((left, top, left + self.size, top + self.size))
        if aug is not None and aug["flip"]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _rgb(self, path, aug=None):
        img = self._fit(Image.open(path).convert("RGB"), Image.BILINEAR, aug)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
        if aug is not None:
            x = x.pow(aug["gamma"]) * aug["bright"]
            gray = x.mean(0, keepdim=True)
            x = gray + (x - gray) * aug["sat"]                 # saturation
            x = x.mean() + (x - x.mean()) * aug["contrast"]    # contrast
            x = x.clamp(0, 1)
        return x

    def _depth(self, path, aug=None):
        if self.mode == "packed_f32":
            # TartanAir V2: RGBA bytes ARE a packed float32 meters value —
            # decode to float BEFORE resizing (resizing the raw RGBA bytes
            # first would corrupt the packing).
            raw = np.asarray(Image.open(path))
            d = raw.view(np.float32).reshape(raw.shape[:2]).copy()
            d[d >= 8000] = 0  # ~8248.1 sentinel = sky / no hit
            img = self._fit(Image.fromarray(d, mode="F"), Image.NEAREST, aug)
            return torch.from_numpy(np.asarray(img).copy()).unsqueeze(0)
        img = self._fit(Image.open(path), Image.NEAREST, aug)
        d = np.asarray(img).astype(np.float32)
        d[d == 65535] = 0  # 16-bit saturation = no reading (vkitti2 sky)
        return (torch.from_numpy(d) / self.scale).unsqueeze(0)

    def __getitem__(self, i, _retries=0, _same_retries=0):
        # "corrupt" reads observed in practice were transient — re-scanning
        # the exact same files outside the DataLoader (no worker contention)
        # found them perfectly readable. So retry the SAME clip a couple
        # times first (cheap, likely just I/O contention from num_workers>1
        # hitting /archive concurrently) before giving up and jumping to a
        # different random clip (in case a file really is dead).
        try:
            seq, start, fs = self.clips[i]
            pairs = seq[start:start + self.T * fs:fs]
            aug = self._draw_aug()  # one draw for the whole clip
            frames = torch.stack([self._rgb(r, aug) for r, _ in pairs])
            depth = torch.stack([self._depth(d, aug) for _, d in pairs])
            valid = (depth > 0).float()
            return frames, depth, valid
        except (OSError, ValueError) as e:
            if _same_retries < 2:
                return self.__getitem__(i, _retries, _same_retries + 1)
            if _retries >= 10:
                raise
            print(f"WARNING: skipping unreadable clip {i} after retries ({e})")
            return self.__getitem__(random.randrange(len(self)), _retries + 1)


class SynthClips(torch.utils.data.IterableDataset):
    """Static depth-gradient background + moving boxes at random depths.
    Mimics the fixed-camera target: most patches static per frame."""

    def __init__(self, size=128, clip_len=4, n_boxes=2):
        self.size, self.T, self.n_boxes = size, clip_len, n_boxes

    def __iter__(self):
        while True:
            S, T = self.size, self.T
            yy = torch.linspace(0.3, 1.0, S).view(S, 1).expand(S, S)
            frames, depths = [], []
            boxes = [(torch.randint(0, S - 32, (2,)).tolist(),
                      torch.randint(-3, 4, (2,)).tolist(),
                      torch.rand(1).item() * 0.25 + 0.05,
                      torch.rand(3).tolist()) for _ in range(self.n_boxes)]
            for t in range(T):
                img = yy.unsqueeze(0).repeat(3, 1, 1) * 0.5
                dep = yy.clone()
                for (pos, vel, d, color) in boxes:
                    y = max(0, min(S - 32, pos[0] + vel[0] * t * 4))
                    x = max(0, min(S - 32, pos[1] + vel[1] * t * 4))
                    img[:, y:y + 32, x:x + 32] = torch.tensor(color).view(3, 1, 1)
                    dep[y:y + 32, x:x + 32] = d
                frames.append(img)
                depths.append(dep.unsqueeze(0))
            f, d = torch.stack(frames), torch.stack(depths)
            yield f, d, torch.ones_like(d)


def build_mixed(specs, holdout=None, val=False, **kw):
    """specs: ["scannet:/path", "folder:/path:2000", ...].
    holdout: list of path substrings (e.g. ["Scene06"]) naming the val
    split; sequences whose paths match go to val. val=False returns the
    train split (matches excluded), val=True the val split (matches only).
    Returns (ConcatDataset, sampler) — sampler equalizes per-dataset draw
    probability so a huge dataset doesn't drown a small one."""
    datasets = []
    for spec in specs:
        parts = spec.split(":")
        name, root = parts[0], parts[1]
        fn, scale, mode = ADAPTERS[name]
        if len(parts) > 2:
            scale = float(parts[2])
        seqs = fn(root)
        if holdout:
            seqs = [s for s in seqs
                    if any(h in s[0][0] for h in holdout) == val]
        ds = ClipDataset(seqs, scale, depth_mode=mode, **kw)
        if len(ds) == 0:
            raise ValueError(f"no clips found for {spec}"
                             + (f" (holdout={holdout}, val={val})" if holdout else ""))
        datasets.append(ds)
    mixed = ConcatDataset(datasets)
    weights = torch.cat([torch.full((len(d),), 1.0 / len(d)) for d in datasets])
    sampler = WeightedRandomSampler(weights, num_samples=len(mixed))
    return mixed, sampler
