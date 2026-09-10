import torch
from sokkanaem.qtemporal_refiner import TemporalErrorRefiner, TemporalRefinedStream, repair_target


def inputs():
    return torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32), torch.rand(1, 1, 32, 32)+1, torch.rand(1, 1, 32, 32)+1


def test_zero_initial_identity_and_bounded_residual():
    m = TemporalErrorRefiner(True); data = inputs()
    pred, gate = m(*data)
    torch.testing.assert_close(pred, data[2], rtol=0, atol=0)
    assert gate.shape == data[2].shape
    with torch.no_grad(): m.fuse[-1].bias.fill_(2.)
    pred, gate = m(*data)
    assert ((pred/data[2]).log().abs() <= .100001).all()
    (pred.mean()+gate.mean()).backward()
    assert m.fuse[-1].weight.grad.abs().sum() > 0


def test_explicit_time_removal_ignores_previous_inputs():
    m = TemporalErrorRefiner(False); a,b,d,p = inputs()
    with torch.no_grad(): m.fuse[-1].weight.normal_(0, .1)
    x,_ = m(a,b,d,p); y,_ = m(a,b+5,d,p+7)
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    m.temporal = True
    x,_ = m(a,b,d,p); y,_ = m(a,b+5,d,p+7)
    assert not torch.equal(x,y)


def test_repair_target_exact_and_wrong_depth():
    d = torch.ones(1,1,16,16)
    assert repair_target(d,d).sum() == 0
    assert repair_target(d,d*2).min() == 1


def test_stream_keeps_refresh_exact_and_caches_independently():
    class Base(torch.nn.Module):
        def step(self, frame, state=None):
            n=0 if state is None else state
            return frame[:,:1]+1,n+1,{"full_refresh":n%2==0,"backbone_calls":int(n%2==0)}
    m = TemporalErrorRefiner(); wrapper = TemporalRefinedStream(Base(), m,.3,150)
    a,b,_,_ = inputs()
    first,s,_ = wrapper.step(a)
    torch.testing.assert_close(first,a[:,:1]+1,rtol=0,atol=0)
    old=s["previous_depth"].clone(); first.add_(10)
    assert torch.equal(s["previous_depth"],old)
    second,s,_=wrapper.step(b,s)
    third,s,info=wrapper.step(a,s)
    assert info["full_refresh"]
    torch.testing.assert_close(third,a[:,:1]+1,rtol=0,atol=0)
