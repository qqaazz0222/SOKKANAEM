"""Stage-two detail-feature experiment with the original bin readout frozen."""
import torch
from torch import nn
import torch.nn.functional as F


class DetailFeaturePath(nn.Module):
    def __init__(self):
        super().__init__()
        self.half_stem = nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.GELU(),
                                  nn.Conv2d(16,16,3,1,1,groups=16),nn.GELU(),
                                  nn.Conv2d(16,16,1),nn.GELU())
        self.quarter = nn.Sequential(nn.Conv2d(16,24,3,2,1),nn.GELU(),
                                     nn.Conv2d(24,24,3,1,1,groups=24),nn.GELU(),
                                     nn.Conv2d(24,16,1),nn.GELU())
        self.semantic_norm = nn.GroupNorm(4,16)
        self.detail_norm = nn.GroupNorm(4,32)
        self.fusion = nn.Sequential(nn.Conv2d(48,16,1),nn.GELU(),nn.Conv2d(16,16,3,1,1))
        nn.init.zeros_(self.fusion[-1].weight);nn.init.zeros_(self.fusion[-1].bias)

    def forward(self,rgb,features):
        fine=self.half_stem(rgb)
        middle=F.interpolate(self.quarter(fine),size=fine.shape[-2:],mode="bilinear",align_corners=False)
        detail=self.detail_norm(torch.cat((fine,middle),1))
        correction=self.fusion(torch.cat((self.semantic_norm(features),detail),1))
        return features + correction
