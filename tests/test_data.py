"""Data pipeline check: fake on-disk dataset -> canonical clips -> train step."""
import numpy as np
import torch
from PIL import Image

from sokkanaem.data import build_mixed


def _fake_dataset(root, n_seq=2, n_frames=8, drop_depth_frame=False):
    for s in range(n_seq):
        rgb = root / f"seq{s}" / "rgb"
        dep = root / f"seq{s}" / "depth"
        rgb.mkdir(parents=True)
        dep.mkdir(parents=True)
        for t in range(n_frames):
            Image.fromarray(
                np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
            ).save(rgb / f"{t:06d}.png")
            d = np.full((48, 64), 3000, dtype=np.uint16)
            d[:8] = 0  # invalid band
            Image.fromarray(d).save(dep / f"{t:06d}.png")


def test_mixed_clips_canonical_format(tmp_path):
    _fake_dataset(tmp_path / "a")
    _fake_dataset(tmp_path / "b")
    mixed, sampler = build_mixed(
        [f"folder:{tmp_path/'a'}", f"tum:{tmp_path/'b'}:2000"],
        clip_len=4, size=64)
    frames, depth, valid = mixed[0]
    assert frames.shape == (4, 3, 64, 64) and frames.max() <= 1.0
    assert depth.shape == (4, 1, 64, 64)
    assert valid.min() == 0 and valid.max() == 1  # invalid band survives resize
    assert abs(depth.max().item() - 3.0) < 1e-5   # folder scale 1000
    # second source uses per-spec scale override (2000 -> 1.5 m)
    b0 = len(mixed.datasets[0])
    _, depth_b, _ = mixed[b0]
    assert abs(depth_b.max().item() - 1.5) < 1e-5
    assert len(list(sampler)) == len(mixed)


def test_vkitti2_layout_and_sky(tmp_path):
    for cam in ("Camera_0", "Camera_1"):
        rgb = tmp_path / "Scene01" / "clone" / "frames" / "rgb" / cam
        dep = tmp_path / "Scene01" / "clone" / "frames" / "depth" / cam
        rgb.mkdir(parents=True)
        dep.mkdir(parents=True)
        for t in range(4):
            Image.fromarray(
                np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
            ).save(rgb / f"rgb_{t:05d}.jpg")
            d = np.full((48, 64), 2000, dtype=np.uint16)  # 20 m
            d[:8] = 65535  # sky
            Image.fromarray(d).save(dep / f"depth_{t:05d}.png")
    mixed, _ = build_mixed([f"vkitti2:{tmp_path}"], clip_len=4, size=64)
    assert len(mixed) == 2  # one clip per camera
    _, depth, valid = mixed[0]
    assert abs(depth.max().item() - 20.0) < 1e-5  # cm scale
    assert valid.min() == 0  # sky marked invalid


def test_holdout_split(tmp_path):
    _fake_dataset(tmp_path / "a", n_seq=3)
    spec = [f"folder:{tmp_path/'a'}"]
    full, _ = build_mixed(spec, clip_len=4, size=64)
    train, _ = build_mixed(spec, clip_len=4, size=64, holdout=["seq1"])
    val, _ = build_mixed(spec, clip_len=4, size=64, holdout=["seq1"], val=True)
    assert len(train) + len(val) == len(full)
    assert len(val) == len(full) // 3  # 1 of 3 identical seqs
    # no path overlap between splits
    train_paths = {p for d in train.datasets for s, *_ in d.clips for p, _ in s}
    val_paths = {p for d in val.datasets for s, *_ in d.clips for p, _ in s}
    assert not (train_paths & val_paths)


def test_aspect_preserving_crop(tmp_path):
    # 128x64 wide frame -> size 64: shorter side fits, width center-cropped
    rgb = tmp_path / "seq0" / "rgb"
    dep = tmp_path / "seq0" / "depth"
    rgb.mkdir(parents=True)
    dep.mkdir(parents=True)
    for t in range(4):
        img = np.zeros((64, 128, 3), dtype=np.uint8)
        img[:, :32] = 255  # left quarter white; must be cropped away
        Image.fromarray(img).save(rgb / f"{t:06d}.png")
        Image.fromarray(np.full((64, 128), 3000, dtype=np.uint16)).save(
            dep / f"{t:06d}.png")
    mixed, _ = build_mixed([f"folder:{tmp_path}"], clip_len=4, size=64)
    frames, depth, _ = mixed[0]
    assert frames.shape == (4, 3, 64, 64) and depth.shape == (4, 1, 64, 64)
    assert frames.max() == 0.0  # center crop of a wide frame drops the left edge


def test_train_step_on_fake_data(tmp_path):
    _fake_dataset(tmp_path / "a")
    from sokkanaem import SOKKANAEM
    from sokkanaem.losses import grad_loss, si_log_loss

    mixed, _ = build_mixed([f"folder:{tmp_path/'a'}"], clip_len=2, size=64)
    frames, gt, valid = mixed[0]
    model = SOKKANAEM()
    depths, masks = model.forward_clip(frames.unsqueeze(0))
    loss = si_log_loss(depths, gt.unsqueeze(0), valid.unsqueeze(0)) \
        + grad_loss(depths, gt.unsqueeze(0), valid.unsqueeze(0))
    loss.backward()
    assert torch.isfinite(loss)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        test_mixed_clips_canonical_format(Path(d) / "x")
        test_vkitti2_layout_and_sky(Path(d) / "v")
        test_train_step_on_fake_data(Path(d) / "y")
    print("data tests passed")
