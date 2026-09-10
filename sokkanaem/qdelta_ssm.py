"""Experimental selective SSM predictor of cached Q0 features between refreshes.

Uses current RGB, cached normalized Q0 features and explicit per-patch state.
Encoder is NOT run on update frames. This approximation needs quality gates.
"""
import torch
from torch import nn
import torch.nn.functional as F
from sokkanaem.ssm import SelectiveSSM


class QDeltaSSM(nn.Module):
    def __init__(self,channels=384,stages=4,width=32,patch=14):
        super().__init__()
        self.channels,self.stages,self.width,self.patch=channels,stages,width,patch
        self.rgb=nn.Conv2d(3,width,patch,stride=patch)
        self.context=nn.Linear(channels,width)
        self.fuse=nn.Sequential(nn.Linear(2*width,width),nn.LayerNorm(width),nn.GELU())
        self.ssm=SelectiveSSM(width,d_state=8,expand=1)
        self.readout=nn.Linear(width,stages*channels)
        self.global_readout=nn.Linear(width,(stages+1)*channels)
        for layer in (self.readout,self.global_readout):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def refresh(self,encoded):
        b,n,_=encoded["features"][-1].shape
        return {"features":[f.detach().clone() for f in encoded["features"]],
                "pooled":encoded["pooled"].detach().clone(),"grid":encoded["grid"],"size":encoded["size"],
                "h":encoded["pooled"].new_zeros(b*(n-1),self.width,8)}

    def update(self,frame,mask,state):
        b,n,c=state["features"][-1].shape
        if b!=1:raise ValueError("Prototype supports one independent stream")
        if mask.shape!=(b,n-1):raise ValueError("Mask must cover Q0 patch tokens, excluding CLS")
        idx=(mask.reshape(-1)>.5).nonzero().flatten()
        if not len(idx):return state,{"updated_patches":0}
        size=(state["grid"][0]*self.patch,state["grid"][1]*self.patch)
        rgb=F.interpolate(frame,size=size,mode="bilinear",align_corners=False)
        local=self.rgb(rgb).flatten(2).transpose(1,2)
        context=self.context(state["features"][-1][:,1:])
        u=self.fuse(torch.cat((local,context),-1)).reshape(-1,self.width)
        # Only active patches enter the SSM. Inactive hidden states are copied
        # bitwise; this implements the binary zero-step update/copy behavior.
        y,h=self.ssm.step(u[idx],h=state["h"][idx])
        hidden=state["h"].clone().index_copy(0,idx,h)
        delta=.2*self.readout(y).tanh().reshape(-1,self.stages,c)
        global_delta=.2*self.global_readout(y.mean(0,keepdim=True)).tanh().reshape(self.stages+1,c)
        features=[]
        for stage,old in enumerate(state["features"]):
            patches=old[:,1:].reshape(-1,c).clone().index_add(0,idx,delta[:,stage])
            cls=old[:,:1]+global_delta[stage][None,None]
            features.append(torch.cat((cls,patches.reshape(b,n-1,c)),1))
        new_state={**state,"features":features,"pooled":state["pooled"]+global_delta[-1][None],"h":hidden}
        return new_state,{"updated_patches":len(idx)}
