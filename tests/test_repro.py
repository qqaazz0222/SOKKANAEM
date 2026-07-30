"""The effective config train.py writes must be readable back by
from_checkpoint(), or every run loses its architecture on reload."""
import argparse
import importlib.util
from pathlib import Path

from sokkanaem import checkpoint_config

spec = importlib.util.spec_from_file_location(
    "train", Path(__file__).resolve().parents[1] / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def test_effective_config_roundtrips(tmp_path):
    args = argparse.Namespace(
        config="configs/main_v8.toml",   # dropped: it is the input, not the fact
        data=["tum:/a", "bonn:/b"], size=256, lr=3e-4, seed=7,
        augment=True, detector_mask=False, teacher_weight=0.0,
        size_schedule=None,              # unset options must not be emitted
        work_dir=str(tmp_path))
    model_kw = {"decoder": "dpt", "bins": 64, "spatial_cache": True,
                "tau_on": 0.05}

    train.write_config(tmp_path / "config.toml", args, model_kw)
    cfg = checkpoint_config(tmp_path / "latest.pt")   # reads the sibling TOML

    assert cfg["model"] == model_kw       # from_checkpoint's kwargs survive
    assert cfg["size"] == 256             # infer/eval restore the train res
    assert cfg["seed"] == 7
    assert cfg["data"] == ["tum:/a", "bonn:/b"]
    assert cfg["augment"] is True and cfg["teacher_weight"] == 0.0
    assert "size_schedule" not in cfg and "config" not in cfg
    assert cfg["meta"]["git_commit"] and cfg["meta"]["torch"]


def test_meta_survives_strict_torch_load(tmp_path):
    """The same meta goes into the checkpoint, and torch.load defaults to
    weights_only=True since 2.6 — a stray non-plain type (torch.__version__ is
    a TorchVersion, a str SUBCLASS) makes every checkpoint of the run
    unloadable, which only shows up after the run finishes."""
    import torch

    meta = train.write_config(tmp_path / "config.toml",
                              argparse.Namespace(config=None, seed=0), {})
    torch.save({"meta": meta}, tmp_path / "ckpt.pt")
    back = torch.load(tmp_path / "ckpt.pt")          # strict default
    assert back["meta"]["torch"] == str(torch.__version__)
