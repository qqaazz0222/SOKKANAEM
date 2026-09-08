"""Versioned evaluation provenance and reserved paper-test identifiers."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess

VERSION = "sokkanaem-paper-v1"
ROOT = Path(__file__).resolve().parents[1]
FINAL_TEST_SEQUENCES = tuple(f"rgbd_dataset_freiburg3_{name}" for name in (
    "sitting_xyz", "sitting_rpy", "walking_xyz", "walking_rpy"))
EVALUATION_FILES = (
    "sokkanaem/__init__.py",
    "scripts/eval_acc.py", "sokkanaem/alignment.py", "sokkanaem/data.py",
    "sokkanaem/metrics.py", "sokkanaem/sharpness.py", "sokkanaem/protocol.py",
    "sokkanaem/model.py", "sokkanaem/qmodel.py", "sokkanaem/ssm.py",
    "sokkanaem/scan_triton.py", "sokkanaem/detector.py", "sokkanaem/gmc.py",
    "paper/evaluation_protocol.json",
)


def runtime_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "numpy", "Pillow", "transformers",
                 "triton", "opencv-python"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def file_record(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def provenance(model, manifest):
    """Hash actual source bytes: a git commit alone misses working-tree edits."""
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True)
    out = {"protocol": VERSION, "git_commit": git.stdout.strip() or "unknown",
           "manifest": file_record(manifest),
           "code": [file_record(ROOT / p) for p in EVALUATION_FILES]}
    kind, name = model.split(":", 1)
    if kind == "ours":
        out["checkpoint"] = file_record(name)
        out["config"] = file_record(Path(name).with_name("config.toml"))
        out["weight_selection"] = "EMA when present, otherwise model state"
    return out


def check_final_test(manifest, final_test=False):
    """A deliberate CLI flag prevents mistaking reserved tests for validation."""
    m = json.loads(Path(manifest).read_text())
    reserved = m.get("role") == "final_test" or any(
        set(Path(p).parts) & set(FINAL_TEST_SEQUENCES)
        for clip in m["clips"] for pair in clip["pairs"] for p in pair)
    if reserved and not final_test:
        raise ValueError("reserved final test: use --final-test only after model selection is frozen")
    if final_test and not reserved:
        raise ValueError("--final-test cannot relabel a legacy validation manifest")
    return "final_test" if reserved else "development_validation"


def guard_training_sequences(sequences):
    reserved = set(FINAL_TEST_SEQUENCES)
    for seq in sequences:
        if seq and set(Path(seq[0][0]).parts) & reserved:
            raise ValueError(f"reserved final-test sequence cannot enter training: {seq[0][0]}")
