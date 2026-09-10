"""RGB-only refresh decision BEFORE flow, with exact same-policy late control."""
import torch
from sokkanaem.qrisk import risk_features
from sokkanaem.qflow import gray_image,correspondence,warp_depth
from sokkanaem.qschedule import ScheduledFlow


def preflow_features(frame,anchor,gap):
    x=risk_features(frame,anchor,gap)
    # Retain identical policy architecture/initialization; six flow-derived
    # coordinates are absent, NOT estimated from current dense Q0 or future data.
    return torch.cat((x,x.new_zeros(len(x),6)),1)


class PreflowScheduledFlow(ScheduledFlow):
    def __init__(self,exact,risk,threshold,reset_h=False,late=False):
        super().__init__(exact,risk,threshold,reset_h);self.late=late

    @torch.no_grad()
    def step(self,frame,state=None):
        gray=gray_image(frame)
        if state is not None and state["depth"].shape[-2:]!=frame.shape[-2:]:state=None
        gap=0 if state is None else state["age"]+1
        force=state is None or gap>=4;predicted=None;incoming=0.;h=None;flow_done=False
        reason="initial" if state is None else "maximum_age"
        candidate=None
        if not force:
            if self.late:
                flow,reliable=correspondence(state["gray"],gray,self.engine)
                candidate,_=warp_depth(state["depth"],flow,reliable,"nearest");flow_done=True
            x=preflow_features(frame,state["anchor"],gap)
            incoming=0. if state["h"] is None or self.reset_h else float(state["h"].abs().mean())
            score,h=self.risk.step(x,state["h"],self.reset_h);predicted=float(score)
            force=gap>=2 and predicted>self.threshold
            reason="risk" if force else ("mandatory_first_skip" if gap==1 else "extended_reuse")
            if not force and candidate is None:
                flow,reliable=correspondence(state["gray"],gray,self.engine)
                candidate,_=warp_depth(state["depth"],flow,reliable,"nearest");flow_done=True
        if force:
            pred,_,_=self.exact.step(frame);age=0;anchor=frame.clone();h=None
        else:pred=candidate;age=gap;anchor=state["anchor"]
        new={"depth":pred.detach().clone(),"gray":gray.copy(),"age":age,"anchor":anchor,
             "h":None if h is None else h.detach().clone()}
        return pred,new,{"full_refresh":force,"backbone_calls":int(force),"cache_age":age,"gap_attempted":gap,
                         "predicted_risk":predicted,"reason":reason,"incoming_h_abs_mean":incoming,
                         "flow_calls":2 if flow_done else 0,"late":self.late}
