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


def test_corrupt_frame_is_skipped_not_fatal(tmp_path):
    # scraped-dataset reality check: a truncated PNG must not kill the run
    _fake_dataset(tmp_path / "a", n_seq=3)
    mixed, _ = build_mixed([f"folder:{tmp_path/'a'}"], clip_len=4, size=64)
    bad_path = mixed.datasets[0].clips[0][0][0][0]
    with open(bad_path, "wb") as f:
        f.write(b"not a real png")
    frames, depth, valid = mixed[0]  # must not raise
    assert frames.shape == (4, 3, 64, 64)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        test_mixed_clips_canonical_format(Path(d) / "x")
        test_vkitti2_layout_and_sky(Path(d) / "v")
        test_train_step_on_fake_data(Path(d) / "y")
    print("data tests passed")


def test_timestamp_pairing_beats_sorted_order(tmp_path):
    """TUM/Bonn: unequal counts + offset capture times. Sorted-order pairing
    misaligns GT; _pair_by_timestamp must match nearest within 0.02s and drop
    rgb frames that have no depth in the window."""
    import os

    from sokkanaem.data import _pair_by_timestamp, _pair_sorted

    rgb, dep = tmp_path / "rgb", tmp_path / "depth"
    rgb.mkdir(), dep.mkdir()
    # depth is missing the 3rd frame and lags rgb by 5ms
    for t in (100.000, 100.033, 100.066, 100.100):
        (rgb / f"{t:.6f}.png").write_bytes(b"")
    for t in (100.005, 100.038, 100.105):
        (dep / f"{t:.6f}.png").write_bytes(b"")

    pairs = _pair_by_timestamp(str(rgb), str(dep))
    got = [(os.path.basename(r), os.path.basename(d)) for r, d in pairs]
    assert got == [("100.000000.png", "100.005000.png"),
                   ("100.033000.png", "100.038000.png"),
                   ("100.100000.png", "100.105000.png")], got
    # the dropped frame is exactly what sorted-order pairing would mismatch
    srt = [(os.path.basename(r), os.path.basename(d))
           for r, d in _pair_sorted(str(rgb), str(dep))]
    assert srt[2] == ("100.066000.png", "100.105000.png"), srt


def test_augmentation_is_clip_consistent_and_geometry_matched(tmp_path):
    """Augmentation must be drawn once per clip (temporal structure survives)
    and applied identically to rgb and depth (geometry stays aligned)."""
    import torch

    # depth encodes the pixel's own column index, so any geometric transform
    # applied to rgb must show up identically in depth
    for s in range(1):
        rgb = tmp_path / f"seq{s}" / "rgb"
        dep = tmp_path / f"seq{s}" / "depth"
        rgb.mkdir(parents=True)
        dep.mkdir(parents=True)
        col = np.tile(np.arange(64, dtype=np.uint16) * 10 + 100, (48, 1))
        for t in range(8):
            img = np.stack([col / 640 * 255] * 3, -1).astype(np.uint8)
            Image.fromarray(img).save(rgb / f"{t:06d}.png")
            Image.fromarray(col).save(dep / f"{t:06d}.png")

    plain, _ = build_mixed([f"folder:{tmp_path}"], clip_len=4, size=64)
    aug, _ = build_mixed([f"folder:{tmp_path}"], clip_len=4, size=64, augment=True)

    f0, d0, _ = plain[0]
    torch.manual_seed(0)
    fa, da, _ = aug[0]
    assert fa.shape == f0.shape and da.shape == d0.shape

    # every frame of the clip got the SAME transform: identical inputs in,
    # identical outputs out (the fake sequence is static)
    assert torch.allclose(fa[0], fa[1]) and torch.allclose(da[0], da[3])

    # rgb and depth stay geometrically aligned: both monotonic in x, and the
    # flip/crop shows in both or neither
    rgb_row = fa[0, 0, 32]
    dep_row = da[0, 0, 32]
    # depth resamples nearest, so it has plateaus and cannot be strictly
    # monotonic — the invariant is that both agree on *direction* (a flip
    # shows in both or neither)
    assert (torch.sign(rgb_row.diff()).sum()
            * torch.sign(dep_row.diff()).sum()) > 0

    # photometric jitter touches rgb only, so depth values stay real metres
    assert 0.05 < da.max().item() < 1.0
