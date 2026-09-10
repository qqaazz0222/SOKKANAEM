"""Q0-faithful streaming foundation and an explicit output-reuse control.

This module does NOT modify qmodel.py or the frozen native paper model.
ChangeAwareQ0 is stage-two control, not yet selective SSM feature updating.
"""
import torch
from torch import nn
import torch.nn.functional as F
from sokkanaem.detector import ChangeDetector


class Q0ExactStream(nn.Module):
    def __init__(self,q0):
        super().__init__()
        if q0.temporal is not None:
            raise ValueError("Use original Q0, not an already trained QT adapter")
        self.q=q0.eval().requires_grad_(False)
        self.backbone_calls=0

    @property
    def infer_size(self): return self.q.infer_size

    @property
    def patch_size(self): return self.q.net.config.patch_size

    @torch.no_grad()
    def encode(self,frame):
        x,size=self.q._prep(frame)
        output=self.q.net.backbone.forward_with_filtered_kwargs(
            x,output_hidden_states=True,output_attentions=False)
        self.backbone_calls+=1
        grid=(x.shape[-2]//self.patch_size,x.shape[-1]//self.patch_size)
        # Decoder consumes layer-normalized feature_maps; calibration consumes
        # RAW final hidden states, exactly as QDepth._shape's original forward.
        raw=output.hidden_states[-1]
        if raw.shape[1]!=grid[0]*grid[1]+1:
            raise ValueError("Unsupported special-token layout; never infer CLS from parity")
        pooled=raw[:,1:].mean(1)
        return {"features":list(output.feature_maps),"pooled":pooled,"grid":grid,"size":size}

    @torch.no_grad()
    def decode(self,encoded):
        disp=self.q._decode(encoded["features"],encoded["grid"])
        return self.q._calibrate(disp,encoded["pooled"],encoded["size"])

    @torch.no_grad()
    def step(self,frame,state=None):
        encoded=self.encode(frame)
        depth=self.decode(encoded)
        return depth,None,{"full_refresh":True,"backbone_calls":1}


class ChangeAwareQ0(nn.Module):
    """Anchor-relative patch detection plus bounded-age WHOLE-output reuse.

    No active-token recomputation is claimed. If a refresh is needed, Q0 runs
    in full. Otherwise the complete previous metric depth is copied.
    All stream state belongs to the caller. Batch size one is deliberate.
    """
    def __init__(self,exact,tau_on=1e-4,tau_off=5e-5,refresh_every=8,max_active=0.,periodic_only=False):
        super().__init__()
        if refresh_every<1 or not 0<=max_active<=1:
            raise ValueError("Invalid refresh policy")
        if tau_off<0 or tau_on<tau_off:raise ValueError("Invalid hysteresis thresholds")
        self.exact=exact
        self.detector=ChangeDetector(patch_size=exact.patch_size,tau_on=tau_on,tau_off=tau_off,
                                    keyframe_every=refresh_every,dilate=True)
        self.refresh_every=refresh_every;self.max_active=max_active;self.periodic_only=periodic_only

    @torch.no_grad()
    def step(self,frame,state=None):
        if frame.ndim!=4 or frame.shape[0]!=1:raise ValueError("One frame from one stream required")
        image=F.interpolate(frame,size=(self.exact.infer_size,self.exact.infer_size),mode="bilinear",align_corners=False)
        if self.exact.infer_size%self.exact.patch_size:
            raise ValueError("Detection grid must match Q0 patches")
        grid=self.exact.infer_size//self.exact.patch_size
        reset=state is None or state["anchor"].shape!=image.shape or state["depth"].shape[-2:]!=frame.shape[-2:]
        if reset:state=None
        age=0 if state is None else state["age"]+1
        force=state is None or age>=self.refresh_every
        det=None if state is None else state["detector"]
        score=None if force else F.avg_pool2d((image-state["anchor"]).square().mean(1,keepdim=True),self.exact.patch_size)[:,0]
        mask,det=self.detector.gate(score,1,grid,grid,frame.device,det)
        active=float(mask.mean())
        # Periodic-only is a negative-control baseline: it ignores all changes.
        refresh=force or (not self.periodic_only and active>self.max_active)
        # Periodic keyframes from the detector also request an exact refresh.
        if not self.periodic_only and self.detector.is_keyframe(None if state is None else state["detector"]):
            refresh=True
        if refresh:
            depth,_,_=self.exact.step(frame)
            next_state={"anchor":image.detach().clone(),"depth":depth.detach().clone(),"age":0,"detector":det}
        else:
            depth=state["depth"].clone()
            next_state={"anchor":state["anchor"],"depth":state["depth"],"age":age,"detector":det}
        return depth,next_state,{"full_refresh":refresh,"backbone_calls":int(refresh),
                                 "active_ratio":active,"cache_age":next_state["age"],
                                 "mode":"whole_output_reuse_control_not_selective_ssm"}
