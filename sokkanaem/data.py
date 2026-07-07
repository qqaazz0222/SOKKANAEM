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
    folder:/data/mine:1000    generic: */rgb/* + */depth/*, custom scale

Metric ranges differ across datasets (indoor ~10m vs KITTI ~80m); mixing is
safe because training uses scale-invariant log loss.
"""
import glob
import os

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
    # ponytail: sorted-order pairing, not timestamp association; use
    # associate.py output layout if frames drop.
    return _subdirs_seqs(root, "rgb", "depth")


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


ADAPTERS = {
    "scannet": (scannet, 1000.0),
    "tum": (tum, 5000.0),
    "bonn": (tum, 5000.0),   # identical layout
    "nyu": (tum, 1000.0),    # extracted-frames layout
    "kitti": (kitti, 256.0),
    "vkitti2": (vkitti2, 100.0),  # depth in cm
    "folder": (tum, 1000.0),  # generic */rgb + */depth, scale via spec
}


class ClipDataset(Dataset):
    """Slices sequences into fixed-length clips; loads and resizes on access."""

    def __init__(self, sequences, depth_scale, clip_len=4, frame_stride=1,
                 clip_stride=2, size=128):
        self.scale = depth_scale
        self.T = clip_len
        self.size = size
        span = clip_len * frame_stride
        self.clips = [(seq, s, frame_stride)
                      for seq in sequences if len(seq) >= span
                      for s in range(0, len(seq) - span + 1, clip_stride)]

    def __len__(self):
        return len(self.clips)

    def _fit(self, img, resample):
        """Aspect-preserving: resize shorter side to self.size, center-crop."""
        w, h = img.size
        s = self.size / min(w, h)
        img = img.resize((max(self.size, round(w * s)),
                          max(self.size, round(h * s))), resample)
        left = (img.width - self.size) // 2
        top = (img.height - self.size) // 2
        return img.crop((left, top, left + self.size, top + self.size))

    def _rgb(self, path):
        img = self._fit(Image.open(path).convert("RGB"), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255

    def _depth(self, path):
        img = self._fit(Image.open(path), Image.NEAREST)
        d = np.asarray(img).astype(np.float32)
        d[d == 65535] = 0  # 16-bit saturation = no reading (vkitti2 sky)
        return (torch.from_numpy(d) / self.scale).unsqueeze(0)

    def __getitem__(self, i):
        seq, start, fs = self.clips[i]
        pairs = seq[start:start + self.T * fs:fs]
        frames = torch.stack([self._rgb(r) for r, _ in pairs])
        depth = torch.stack([self._depth(d) for _, d in pairs])
        valid = (depth > 0).float()
        return frames, depth, valid


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
        fn, scale = ADAPTERS[name]
        if len(parts) > 2:
            scale = float(parts[2])
        seqs = fn(root)
        if holdout:
            seqs = [s for s in seqs
                    if any(h in s[0][0] for h in holdout) == val]
        ds = ClipDataset(seqs, scale, **kw)
        if len(ds) == 0:
            raise ValueError(f"no clips found for {spec}"
                             + (f" (holdout={holdout}, val={val})" if holdout else ""))
        datasets.append(ds)
    mixed = ConcatDataset(datasets)
    weights = torch.cat([torch.full((len(d),), 1.0 / len(d)) for d in datasets])
    sampler = WeightedRandomSampler(weights, num_samples=len(mixed))
    return mixed, sampler
