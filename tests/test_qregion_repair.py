import torch
from sokkanaem.qregion_repair import RegionRepair,no_harm_loss


def test_initial_identity_and_exact_support_after_nonzero_head():
    m=RegionRepair();rgb=torch.rand(1,3,24,24);coarse=torch.rand(1,1,24,24)+1
    mask=torch.zeros_like(coarse,dtype=torch.bool);mask[...,6:18,6:18]=True
    assert torch.equal(m(rgb,rgb,coarse,mask),coarse)
    with torch.no_grad():m.fuse[-1].bias.fill_(1.)
    out=m(rgb,rgb,coarse,mask)
    assert torch.equal(out[~mask],coarse[~mask]) and (out[mask]>coarse[mask]).all()
    assert (out/coarse).log().abs().max() <= .030001
    m.local=False; assert (m(rgb,rgb,coarse,mask)[~mask]>coarse[~mask]).all()


def test_empty_mask_identity_and_finite_gradient():
    m=RegionRepair();rgb=torch.rand(1,3,24,24);coarse=torch.ones(1,1,24,24)
    out=m(rgb,rgb,coarse,torch.zeros_like(coarse,dtype=torch.bool))
    assert torch.equal(out,coarse)
    out.sum().backward();assert torch.isfinite(m.fuse[-1].weight.grad).all()


def test_no_harm_zero_for_baseline_positive_for_error_increase():
    base=torch.ones(1,1,24,24);teacher=base.clone()
    a,b=no_harm_loss(base,base,teacher);assert a==0 and b==0
    pred=base.clone();pred[...,12:]=2
    a,b=no_harm_loss(pred,base,teacher);assert a>0 and b>0
