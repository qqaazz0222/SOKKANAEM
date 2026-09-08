import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "quality_screen", Path(__file__).resolve().parents[1] / "scripts/quality_refinement.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def sample(**overrides):
    values = dict(absrel_raw=.1, absrel_clip_median=.1, boundary_f1=.4,
                  overshoot=.2, flat_tv=.03, grad_ratio=.5)
    values.update(overrides)
    return {"balanced": values, "sources": {"tum": values, "bonn": values}}


def test_gate_rejects_noise_and_oversharpening():
    base = sample()
    assert not screen.gate(base, base)["pass"]
    assert screen.gate(base, sample(boundary_f1=.41))["pass"]
    assert not screen.gate(base, sample(boundary_f1=.41, overshoot=.201))["pass"]
    assert not screen.gate(base, sample(boundary_f1=.41, flat_tv=.031))["pass"]
    assert not screen.gate(base, sample(boundary_f1=.41, grad_ratio=1.01))["pass"]
    assert not screen.gate(base, sample(boundary_f1=.41, absrel_raw=.101))["pass"]


def test_aggregation_balances_sources_and_scenes():
    rows = [{"source": "a", "scene": "a1", "metrics": {"x": 0.}}] * 9
    rows += [{"source": "a", "scene": "a2", "metrics": {"x": 2.}},
             {"source": "b", "scene": "b1", "metrics": {"x": 5.}}]
    assert screen.aggregate(rows)["balanced"]["x"] == 3.
