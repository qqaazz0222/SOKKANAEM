import numpy as np
import torch
from sokkanaem.qpreflow import preflow_features,PreflowScheduledFlow
from sokkanaem.qrisk import risk_features


def test_rgb_only_features_exact_prefix_and_zero_flow_coordinates():
    a=torch.rand(1,3,64,64);b=torch.rand_like(a)
    x=preflow_features(a,b,2)
    assert torch.equal(x[:,:14],risk_features(a,b,2)) and not x[:,14:].any()


def test_predecision_avoids_rejected_flow_without_changing_output_or_schedule():
    class Exact:
        def step(self,frame):return frame[:,:1]+1,None,{}
    class Constant:
        def step(self,x,h=None,reset_h=False):return x.new_ones(1,1),x.new_ones(1,16,4)
    class ZeroFlow:
        def __init__(self):self.calls=0
        def calc(self,a,b,flow):self.calls+=1;return np.zeros((*a.shape,2),np.float32)
    frame=torch.rand(1,3,64,64);results=[]
    for late in (False,True):
        m=PreflowScheduledFlow(Exact(),Constant(),0,late=late);m.engine=ZeroFlow();state=None;rows=[]
        for _ in range(9):
            pred,state,info=m.step(frame,state);rows.append((pred,info["full_refresh"]))
            if info["reason"]=="risk":assert info["flow_calls"]==(2 if late else 0)
        results.append(rows);assert m.engine.calls==(16 if late else 8)
    for (a,fa),(b,fb) in zip(*results):assert fa==fb and torch.equal(a,b)
