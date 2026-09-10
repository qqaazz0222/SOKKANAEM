import numpy as np
import torch
from sokkanaem.qaligned_adapter import AlignedAdapter
from sokkanaem.qrecurrent_flow import advance, K4OutputFlow


def test_history_is_used_only_after_first_update_and_transition_has_gradient():
    torch.manual_seed(739); model = AlignedAdapter("ssm")
    with torch.no_grad(): model.readout.weight.normal_(std=.01)
    encoded = {"features": [torch.randn(1,5,384) for _ in range(4)], "pooled": torch.ones(1,384), "grid": (2,2), "size": (28,28)}
    initial = model.refresh(encoded); frame = torch.rand(1,3,28,28); mask = torch.ones(1,4)
    flow = np.zeros((8,8,2), np.float32); reliable = np.ones((8,8), bool)
    carry, _ = advance(model, initial, frame, mask, flow, reliable)
    reset, _ = advance(model, initial, frame, mask, flow, reliable, reset_h=True)
    for a,b in zip(carry["features"],reset["features"]): assert torch.equal(a,b)
    old_h = carry["h"].clone()
    next_carry, info = advance(model, carry, frame, mask, flow, reliable)
    next_reset, ri = advance(model, carry, frame, mask, flow, reliable, reset_h=True)
    assert info["incoming_h_abs_mean"] > 0 and ri["incoming_h_abs_mean"] == 0
    assert torch.equal(carry["h"],old_h)
    assert any(not torch.equal(a,b) for a,b in zip(next_carry["features"],next_reset["features"]))
    sum(f.square().mean() for f in next_carry["features"]).backward()
    assert model.ssm.A_log.grad.abs().max() > 0 and torch.isfinite(model.ssm.A_log.grad).all()


def test_output_control_k4_schedule_exact_refresh_and_cache_ownership():
    class Exact:
        def step(self, frame): return frame[:,:1]+1, None, {}
    model = K4OutputFlow(Exact()); state = None; flags = []
    frame = torch.rand(1,3,64,64)
    for t in range(9):
        pred,state,info = model.step(frame,state); flags.append(info["full_refresh"])
        assert torch.equal(pred,frame[:,:1]+1)
        assert state["depth"].data_ptr() != pred.data_ptr()
    assert flags == [True,False,False,False,True,False,False,False,True]
