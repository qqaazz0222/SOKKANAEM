"""Learned K2-to-K4 whole-frame refresh scheduling, no depth correction.

Every full refresh is followed by at least one flow-only output; a learned
policy can extend reuse to age2/3, then age4 forces Q0. No quality guarantee.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sokkanaem.qrisk import risk_features
from sokkanaem.ssm import SelectiveSSM
from sokkanaem.qflow import gray_image,correspondence,warp_depth


def scheduling_features(frame,anchor,gap,coarse,warped_rgb,reliable,flow):
    basic=risk_features(frame,anchor,gap)
    magnitude=np.linalg.norm(flow,axis=-1)
    logdepth=coarse.clamp_min(1e-4).log()
    gradient=(torch.diff(logdepth,dim=-1).abs().mean()+torch.diff(logdepth,dim=-2).abs().mean())/2
    extra=torch.stack((1-reliable.float().mean(),basic.new_tensor(float(magnitude.mean())/128),
                       basic.new_tensor(float(np.quantile(magnitude,.95))/128),(frame-warped_rgb).abs().mean(),
                       gradient,logdepth.std(unbiased=False)))[None]
    return torch.cat((basic,extra),1)


class PolicyMLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(16,39),nn.GELU(),nn.Linear(39,16))
    def step(self,u,h=None):return self.net(u),torch.zeros_like(h)


class ScheduleRisk(nn.Module):
    def __init__(self,mode="ssm"):
        super().__init__()
        if mode not in ("ssm","mlp"):raise ValueError("Unsupported policy")
        self.input=nn.Sequential(nn.Linear(20,16),nn.LayerNorm(16),nn.GELU())
        self.core=SelectiveSSM(16,d_state=4,expand=1)
        self.head=nn.Linear(16,1)
        if mode=="mlp":self.core=PolicyMLP()
        self.register_buffer("center",torch.zeros(20));self.register_buffer("scale",torch.ones(20))
        self.mode=mode

    def step(self,x,h=None,reset_h=False):
        if h is None or reset_h:h=x.new_zeros(len(x),16,4)
        y,new_h=self.core.step(self.input((x-self.center)/self.scale),h=h)
        return .05*F.softplus(self.head(y)),new_h


class ScheduledFlow(nn.Module):
    def __init__(self,exact,risk,threshold,reset_h=False):
        super().__init__()
        if threshold<0:raise ValueError("Nonnegative risk threshold required")
        self.exact=exact;self.risk=risk;self.threshold=threshold;self.reset_h=reset_h
        self.engine=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

    @torch.no_grad()
    def step(self,frame,state=None):
        gray=gray_image(frame)
        if state is not None and state["depth"].shape[-2:]!=frame.shape[-2:]:state=None
        gap=0 if state is None else state["age"]+1
        force=state is None or gap>=4;predicted=None;incoming=0.;h=None
        reason="initial" if state is None else "maximum_age"
        if not force:
            flow,reliable=correspondence(state["gray"],gray,self.engine)
            moved,mask=warp_depth(torch.cat((state["depth"],state["rgb"]),1),flow,reliable,"nearest")
            candidate=moved[:,:1]
            x=scheduling_features(frame,state["anchor"],gap,candidate,moved[:,1:],mask,flow)
            incoming=0. if state["h"] is None or self.reset_h else float(state["h"].abs().mean())
            score,h=self.risk.step(x,state["h"],self.reset_h);predicted=float(score)
            force=gap>=2 and predicted>self.threshold
            reason="risk" if force else ("mandatory_first_skip" if gap==1 else "extended_reuse")
        if force:
            pred,_,_=self.exact.step(frame);age=0;anchor=frame.clone();h=None
        else:
            pred=candidate;age=gap;anchor=state["anchor"]
        next_state={"depth":pred.detach().clone(),"rgb":frame.clone(),"gray":gray.copy(),"age":age,
                    "anchor":anchor,"h":None if h is None else h.detach().clone()}
        return pred,next_state,{"full_refresh":force,"backbone_calls":int(force),"cache_age":age,"gap_attempted":gap,
                                "predicted_risk":predicted,"reason":reason,"incoming_h_abs_mean":incoming}
