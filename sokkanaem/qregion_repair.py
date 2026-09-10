"""Bounded RGB repair on uncertain correspondence regions of passing K2 flow.

Mask is an RGB heuristic, not disocclusion truth. The CNN is dense: restricting
the edited OUTPUT is not a claim of sparse neural computation or SSM novelty.
"""
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from sokkanaem.qflow import gray_image, correspondence, warp_depth


class RegionRepair(nn.Module):
    def __init__(self, local=True, bound=.03):
        super().__init__(); self.local = local; self.bound = bound
        self.shallow = nn.Sequential(nn.Conv2d(11,16,3,padding=1),nn.GELU())
        self.context = nn.Sequential(nn.Conv2d(16,32,3,stride=2,padding=1),nn.GELU(),
                                     nn.Conv2d(32,32,3,padding=2,dilation=2),nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(48,16,3,padding=1),nn.GELU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.fuse[-1].weight); nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, current, warped_rgb, coarse, editable):
        mask = editable.to(coarse.dtype)
        x = torch.cat((current,warped_rgb,(current-warped_rgb).abs(),coarse.clamp_min(1e-4).log(),mask),1)
        local = self.shallow(x)
        context = F.interpolate(self.context(local),coarse.shape[-2:],mode="bilinear",align_corners=False)
        raw = self.fuse(torch.cat((local,context),1))
        # Attenuate correction near support boundary without swapping two depth maps.
        weight = F.avg_pool2d(mask,5,stride=1,padding=2)*mask if self.local else torch.ones_like(mask)
        candidate = coarse*(self.bound*raw.tanh()*weight).exp()
        return torch.where(editable,candidate,coarse) if self.local else candidate


def no_harm_loss(pred, base, teacher):
    """Training-only penalty for increasing Q0 errors; no inference guarantee."""
    p,b,t = (x.clamp_min(1e-4).log() for x in (pred,base.detach(),teacher.detach()))
    depth = ((p-t).abs()-(b-t).abs()).relu().mean()
    terms=[]
    for stride in (1,2,4):
        a,c,d = (x[...,::stride,::stride] for x in (p,b,t))
        for dim in (-1,-2):
            da,dc,dd = (torch.diff(x,dim=dim) for x in (a,c,d))
            terms.append(((da-dd).abs()-(dc-dd).abs()).relu().mean())
    return depth,torch.stack(terms).mean()


class RegionFlowStream(nn.Module):
    def __init__(self, exact, repair):
        super().__init__(); self.exact = exact; self.repair = repair
        self.engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

    @torch.no_grad()
    def step(self, frame, state=None):
        gray = gray_image(frame)
        if state is not None and state["depth"].shape[-2:] != frame.shape[-2:]: state = None
        force = state is None or state["age"]>=1
        coverage = None; outside = None
        if force:
            pred,_,_ = self.exact.step(frame); age = 0
        else:
            flow,reliable = correspondence(state["gray"],gray,self.engine)
            transported,mask = warp_depth(torch.cat((state["depth"],state["rgb"]),1),flow,reliable,"nearest")
            coarse,warped_rgb = transported[:,:1],transported[:,1:]
            editable = ~mask
            pred = self.repair(frame,warped_rgb,coarse,editable)
            pred = pred.clamp(min=self.exact.q.d_min,max=self.exact.q.d_max)
            if self.repair.local: pred = torch.where(editable,pred,coarse)
            coverage = float(editable.float().mean())
            outside = float((pred-coarse).abs().masked_fill(editable,0).max())
            age = state["age"]+1
        return pred,{"depth":pred.detach().clone(),"rgb":frame.clone(),"gray":gray.copy(),"age":age},{
            "full_refresh":force,"backbone_calls":int(force),"editable_fraction":coverage,
            "outside_edit_max_abs":outside,"local":self.repair.local,"cache_age":age}
