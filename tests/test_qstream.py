import pytest
import torch
from sokkanaem.qstream import ChangeAwareQ0


class ExactStub(torch.nn.Module):
    infer_size=28
    patch_size=14
    def __init__(self):
        super().__init__();self.calls=0
    def step(self,frame,state=None):
        self.calls+=1
        return frame.mean(1,keepdim=True)+1,None,{}


def test_identical_frame_reuse_and_forced_refresh():
    exact=ExactStub();m=ChangeAwareQ0(exact,refresh_every=4)
    frame=torch.zeros(1,3,28,28);state=None;flags=[]
    for _ in range(9):
        p,state,info=m.step(frame,state);flags.append(info["full_refresh"])
        assert torch.equal(p,torch.ones_like(p))
    assert flags==[True,False,False,False,True,False,False,False,True]
    assert exact.calls==3


def test_anchor_detects_accumulated_drift_and_scene_change():
    m=ChangeAwareQ0(ExactStub(),tau_on=.001,tau_off=.0005,refresh_every=100)
    state=None
    for value,expected in [(0.,True),(.02,False),(.04,True),(.5,True)]:
        _,state,info=m.step(torch.full((1,3,28,28),value),state)
        assert info["full_refresh"]==expected


def test_stream_isolation_and_cached_output_not_mutable():
    m=ChangeAwareQ0(ExactStub())
    p,a,_=m.step(torch.zeros(1,3,28,28));_,b,_=m.step(torch.ones(1,3,28,28))
    p.fill_(99)
    p,a,info=m.step(torch.zeros(1,3,28,28),a)
    assert p.max()==1 and not info["full_refresh"]
    p,b,info=m.step(torch.ones(1,3,28,28),b)
    assert p.max()==2 and not info["full_refresh"]


def test_periodic_control_and_bad_config():
    m=ChangeAwareQ0(ExactStub(),refresh_every=4,periodic_only=True)
    _,state,_=m.step(torch.zeros(1,3,28,28))
    pred,_,info=m.step(torch.ones(1,3,28,28),state)
    assert pred.max()==1 and not info["full_refresh"]
    with pytest.raises(ValueError):ChangeAwareQ0(ExactStub(),refresh_every=0)
