import numpy as np
import torch
from sokkanaem.qflow_features import transport_features


def state():
    f = torch.arange(34).float().reshape(1, 17, 2)
    return {"features": [f, f+100], "pooled": torch.ones(1, 2), "grid": (4, 4), "size": (16, 16),
            "h": torch.arange(32).float().reshape(16, 2, 1)}


def test_feature_identity_and_caller_nonmutation():
    s = state(); old = [f.clone() for f in s["features"]]
    out, changed = transport_features(s, np.zeros((8, 8, 2), np.float32), np.ones((8, 8), bool))
    assert changed == 0 and torch.equal(out["h"], s["h"])
    for before, after, original in zip(old, out["features"], s["features"]):
        assert torch.equal(before, after) and torch.equal(before, original)


def test_feature_hidden_transport_same_direction_globals_unchanged():
    s = state(); flow = np.zeros((8, 8, 2), np.float32); flow[..., 0] = 2
    out, changed = transport_features(s, flow, np.ones((8, 8), bool))
    assert changed == .75 and out["pooled"] is s["pooled"]
    for a, b in zip(s["features"], out["features"]):
        assert torch.equal(a[:, :1], b[:, :1])
        assert torch.equal(a[:, 1:].reshape(4, 4, 2)[:, 1:], b[:, 1:].reshape(4, 4, 2)[:, :-1])
    assert torch.equal(s["h"].reshape(4, 4, 2)[:, 1:], out["h"].reshape(4, 4, 2)[:, :-1])
