"""Paper-preparation regressions; synthetic data only, no final-test inference."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from sokkanaem.data import build_mixed, load_manifest
from sokkanaem.protocol import (FINAL_TEST_SEQUENCES, VERSION, check_final_test,
                                file_record, guard_training_sequences)


def manifest(tmp_path):
    clips = []
    for i in range(2):
        seq = tmp_path / f"seq{i}"
        (seq / "rgb").mkdir(parents=True)
        (seq / "depth").mkdir()
        pairs = []
        for t in range(2):
            rgb, dep = seq / "rgb" / f"{t}.png", seq / "depth" / f"{t}.png"
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(rgb)
            Image.fromarray(np.full((16, 16), 2000 if i == 0 else 0,
                                    dtype=np.uint16)).save(dep)
            pairs.append([str(rgb), str(dep)])
        # Intentionally stale valid_px: runtime GT, not metadata, determines IDs.
        clips.append({"source": "fake", "sequence": seq.name,
                      "pairs": pairs, "valid_px": 0 if i == 0 else 999})
    doc = {"clip_len": 2, "size": 16,
           "sources": {"fake": {"scale": 1000.0, "mode": "u16"}}, "clips": clips}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(doc))
    return path, doc


def test_final_test_requires_deliberate_flag(tmp_path):
    path, doc = manifest(tmp_path)
    assert check_final_test(path) == "development_validation"
    with pytest.raises(ValueError, match="cannot relabel"):
        check_final_test(path, True)
    doc["role"] = "final_test"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="reserved final test"):
        check_final_test(path)
    assert check_final_test(path, True) == "final_test"
    del doc["role"]
    doc["clips"][0]["pairs"][0][0] = f"/data/{FINAL_TEST_SEQUENCES[0]}/rgb/0.png"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="reserved final test"):
        check_final_test(path)


def test_training_reservation_matches_whole_path_components(tmp_path):
    reserved = FINAL_TEST_SEQUENCES[0]
    with pytest.raises(ValueError, match="cannot enter training"):
        guard_training_sequences([[ (f"/data/{reserved}/rgb/0.png", "depth.png") ]])
    guard_training_sequences([[ (f"/data/{reserved}_different/rgb/0.png", "depth.png") ]])


@pytest.mark.parametrize("val", [False, True])
def test_mixed_builder_cannot_load_reserved_sequences(tmp_path, val):
    root = tmp_path / FINAL_TEST_SEQUENCES[0]
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    for kind in ("rgb", "depth"):
        (root / kind / "0.png").write_bytes(b"unused")
    with pytest.raises(ValueError, match="reserved final-test sequence"):
        build_mixed([f"folder:{tmp_path}"], clip_len=1, val=val)


def test_sealed_manifest_does_not_replace_corrupt_clip(tmp_path, monkeypatch):
    path, doc = manifest(tmp_path)
    ds = load_manifest(path)[0][1]
    Path(doc["clips"][0]["pairs"][0][0]).write_bytes(b"corrupt png")
    import sokkanaem.data as data
    def no_random(*args, **kwargs):
        pytest.fail("sealed evaluation must not substitute another clip")
    monkeypatch.setattr(data.random, "randrange", no_random)
    with pytest.raises((OSError, ValueError)):
        ds[0]


def test_manifest_length_is_validated(tmp_path):
    path, doc = manifest(tmp_path)
    doc["clips"][0]["pairs"].pop()
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="length/source mismatch"):
        load_manifest(path)


def test_metric_and_scaled_panels_end_to_end(tmp_path, monkeypatch):
    from scripts import eval_acc
    path, doc = manifest(tmp_path)
    ckpt = tmp_path / "latest.pt"
    ckpt.write_bytes(b"mock runner weights")
    ckpt.with_name("config.toml").write_text("size = 16\n")
    called = []
    def runner(*args, **kwargs):
        def run(frames):
            called.append(True)
            return torch.full((2, 1, 16, 16), 4.0), {"active": .25, "active_all": .625}
        return run, "depth", torch.nn.Linear(1, 1), {"size": (16, 16)}
    monkeypatch.setattr(eval_acc, "ours_runner", runner)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["eval_acc.py", "--manifest", str(path),
        "--model", f"ours:{ckpt}", "--align", "none", "--align", "median",
        "--out-dir", str(out), "--tag", "check", "--reps", "20"])
    eval_acc.main()
    result = json.loads((out / "check.json").read_text())
    assert len(called) == 1 and result["no_gt"] == 1
    assert result["pooled"]["none"]["fake"]["absrel"] == 1.0
    assert result["pooled"]["median"]["fake"]["absrel"] == 0.0
    assert result["protocol_version"] == VERSION
    assert result["dataset_role"] == "development_validation"
    assert result["provenance"]["checkpoint"]["sha256"] == file_record(ckpt)["sha256"]
    assert result["clip_ids"]["fake"] == [{"manifest_index_within_source": 0,
        "sequence": "seq0", "first_rgb": doc["clips"][0]["pairs"][0][0]}]
    assert result["effective_size"] == [16, 16]
    assert result["active"] == .25 and result["active_all_frames"] == .625


def test_sealed_write_is_idempotent_and_refuses_changes(tmp_path):
    from scripts.prepare_paper_test import sealed_write
    path = tmp_path / "seal.json"
    sealed_write(path, {"a": 1})
    original = path.read_bytes()
    sealed_write(path, {"a": 1})
    with pytest.raises(RuntimeError, match="refusing to change"):
        sealed_write(path, {"a": 2})
    assert path.read_bytes() == original
