import torch
from sokkanaem.qschedule import ScheduleRisk,ScheduledFlow


def test_matched_common_initialization_and_core_budget():
    torch.manual_seed(741);a=ScheduleRisk()
    torch.manual_seed(741);b=ScheduleRisk("mlp")
    for name in ("input","head"):
        for x,y in zip(getattr(a,name).parameters(),getattr(b,name).parameters()):assert torch.equal(x,y)
    assert sum(p.numel() for p in a.parameters())==1689
    assert sum(p.numel() for p in b.parameters())==1688


def test_history_changes_scores_and_reset_is_exact():
    m=ScheduleRisk();x=torch.randn(2,20)
    a,h=m.step(x);b,_=m.step(x,h);c,_=m.step(x,h,True)
    assert torch.equal(a,c) and not torch.equal(a,b)
    b.sum().backward();assert m.core.A_log.grad.abs().max()>0


def test_threshold_extremes_recover_k2_k4_and_caller_state_ownership():
    class Exact:
        def step(self,frame):return frame[:,:1]+1,None,{}
    class Constant:
        def step(self,x,h=None,reset_h=False):return x.new_ones(1,1),x.new_ones(1,16,4)
    frame=torch.rand(1,3,64,64)
    for threshold,expected in ((0,[True,False,True,False,True,False,True,False,True]),
                               (2,[True,False,False,False,True,False,False,False,True])):
        model=ScheduledFlow(Exact(),Constant(),threshold);state=None;flags=[]
        for _ in range(9):
            old=None if state is None else state["depth"].clone()
            pred,new,info=model.step(frame,state);flags.append(info["full_refresh"])
            assert torch.equal(pred,frame[:,:1]+1)
            if old is not None:assert torch.equal(state["depth"],old)
            assert new["depth"].data_ptr()!=pred.data_ptr();state=new
        assert flags==expected
