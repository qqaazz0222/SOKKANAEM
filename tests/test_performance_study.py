import copy

import numpy as np
from PIL import Image
import pytest
import torch

from sokkanaem import SOKKANAEM
from sokkanaem import performance_study as perf
from sokkanaem.data import ClipDataset
from sokkanaem.study import native_predict, original_views


def test_common_rgb_is_exact_frozen_loader_and_original_roi_matches(tmp_path):
    path = tmp_path / "rgb.png"
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 256, (481, 639, 3), dtype=np.uint8)).save(path)
    with Image.open(path) as image:
        view = perf.rgb_view(image)
        original = perf.rgb_view(image, original=True)
    tensor = torch.from_numpy(np.asarray(view).copy()).permute(2, 0, 1).float() / 255
    assert torch.equal(tensor, ClipDataset([], 1000, size=256)._rgb(str(path)))
    assert np.array_equal(original, original_views([[str(path), "unused"]])[0])


def test_persistent_memory_traverses_detector_and_deduplicates_aliases():
    x = torch.ones(100)
    state = {"hs": [x], "cache": x.view(10, 10), "det": {"prev": x[:5], "mask": torch.ones(4)}}
    state["cycle"] = state
    assert perf.persistent_bytes(state) == (100 + 4) * 4


@pytest.fixture
def native_factory(monkeypatch):
    torch.manual_seed(3)
    base = SOKKANAEM(dim=32, depth=2, d_state=4, spatial_cache=True, temporal_cache=True,
                    tau_on=.05, tau_off=.025, keyframe_every=3).eval()
    def factory(checkpoint, device, **kwargs):
        model = copy.deepcopy(base).to(device)
        for name in ("spatial_cache", "temporal_cache"):
            if name in kwargs:
                setattr(model, name, kwargs[name])
        return model
    monkeypatch.setattr(perf, "from_checkpoint", factory)
    return base


@pytest.mark.parametrize("spec,reference", [
    ({"kind": "native"}, {}),
    ({"kind": "native", "dense": True}, {"dense": True}),
    ({"kind": "native", "dense": True, "reset": True}, {"reset": True}),
    ({"kind": "native", "hold": True}, {"output_hold": True}),
])
def test_real_streaming_adapter_matches_previous_intervention(native_factory, spec, reference):
    frames = torch.rand(6, 3, 32, 32)
    frames[1:3] = frames[0]
    frames[4:] = frames[3]
    base = native_factory
    _, masks, _ = native_predict(base, frames, {})
    if reference.get("output_hold"):
        base.spatial_cache = base.temporal_cache = False
    expected, _, _ = native_predict(base, frames, reference, masks)
    model = perf.Native(spec, device="cpu")
    with torch.no_grad():
        actual = torch.cat([model.infer(f[None])[0] for f in frames])
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)
    if spec.get("reset"):
        assert perf.persistent_bytes(model.state) == 0


def test_profiling_does_not_change_native_prediction(native_factory):
    frame = torch.rand(1, 3, 32, 32)
    normal = perf.Native({"kind": "native"}, device="cpu")
    timer = perf.StageTimer("cpu")
    instrumented = perf.Native({"kind": "native"}, device="cpu", timer=timer)
    with torch.no_grad():
        assert torch.equal(normal.infer(frame)[0], instrumented.infer(frame)[0])
    assert set(timer.values) == {"detector", "embedding", "backbone", "decoder"}
    assert all(v >= 0 for v in timer.values.values())


def test_four_sequence_inference_cannot_manufacture_five_percent_significance():
    result = perf.paired_sequence_summary([1, 1, 1, 1])
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 1
    assert result["sign_flip_p"] == .125
    assert result["bootstrap_draws"] == 256 and result["sign_flip_draws"] == 16
    assert perf.paired_sequence_summary([0, 0, 0, 0])["sign_flip_p"] == 1


def test_paired_sequence_reversal_reverses_interval():
    a = perf.paired_sequence_summary([-.1, .2, .4, -.7])
    b = perf.paired_sequence_summary([.1, -.2, -.4, .7])
    assert a["difference"] == pytest.approx(-b["difference"])
    assert a["ci_low"] == pytest.approx(-b["ci_high"])
    assert a["sign_flip_p"] == b["sign_flip_p"]
    with pytest.raises(ValueError):
        perf.paired_sequence_summary([np.nan, 0])


def test_pareto_retains_ties_and_rejects_dominated_points():
    assert perf.pareto_flags([(1, 2), (2, 1), (2, 2), (1, 2)]) == [True, True, False, True]


def test_quality_aggregation_distinguishes_source_and_sequence_estimands():
    from scripts.paper_efficiency_report import aggregate_quality
    def row(source, sequence, error):
        sums = {"rel_sum": error, "sq_sum": error**2, "d1_sum": 0., "px": 1.,
                "td_sum": 0., "td_px": 1., "opw_sum": 0., "tce_sum": 0., "warp_px": 1.}
        return {"source": source, "sequence": sequence, "models": {"m": {"gauges": {"none": {
            "scores": {"_pooled": sums}, "failed": error > 1, "catastrophic": error > 1}}}}}
    records = [row("a", "a1", 4), row("b", "b1", 0), row("b", "b2", 0), row("b", "b3", 0)]
    item = aggregate_quality(records)[("m", "none")]
    assert item["balanced"]["absrel"] == 2
    assert item["equal_sequence"]["absrel"] == 1
    assert item["failed"] == 1 and len(item["sequences"]) == 4
    # Doubling clips of the sole a sequence does not create an extra cluster.
    item = aggregate_quality(records + records[:1])[("m", "none")]
    assert item["equal_sequence"]["absrel"] == 1 and len(item["sequences"]) == 4


def test_timing_summary_keeps_all_repeats_not_fastest():
    from scripts.paper_efficiency_report import latency_summary
    telemetry = [{"state_bytes": 4, "keyframe": False, "active": None,
                  "dense_fallback": False, "gmc_calls": 0, "gmc_fallbacks": 0}] * 2
    row = {"model": "midas_256", "source": "a", "sequence": "seq1", "clip_id": "a:0", "frames": 2,
           "runs": [{"ms": times, "telemetry": telemetry, "peak_allocated_mib": 1} for times in ([1, 1], [3, 3])]}
    seq, total, events = latency_summary([row])
    assert total["midas_256"]["mean_ms"] == 2
    assert total["midas_256"]["repeat_min_ms"] == 1
    assert total["midas_256"]["repeat_max_ms"] == 3
    assert events[0]["unique_frames"] == 2 and events[0]["timed_frames_with_repeats"] == 4


def test_scoring_flow_never_builds_an_autograd_graph(monkeypatch):
    from scripts.paper_efficiency_study import score_setup
    from sokkanaem import metrics
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    observed = []
    def flow(a, b, **kwargs):
        observed.append(torch.is_grad_enabled())
        return torch.zeros(len(a), 2, *a.shape[-2:])
    monkeypatch.setattr(metrics, "_flow", flow)
    dataset = [(torch.ones(3, 3, 16, 16), torch.ones(3, 1, 16, 16), torch.ones(3, 1, 16, 16))]
    with torch.enable_grad():
        assert score_setup(dataset, 0) is not None
    assert observed == [False]
