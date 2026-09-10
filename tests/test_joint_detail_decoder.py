import torch
from sokkanaem.model import DPTDecoder
from sokkanaem.joint_detail_decoder import JointDetailDecoder, reliable_edge_loss


def decoder():
    d=DPTDecoder(16,bins=8); d.set_n_blocks(2)
    return d


def test_joint_identity_and_frozen_source():
    torch.manual_seed(1)
    d=decoder().eval().requires_grad_(False)
    rgb=torch.rand(2,3,32,32); tokens=[torch.rand(2,16,2,2) for _ in range(2)]
    model=JointDetailDecoder(d).eval()
    p=model(tokens,rgb)
    torch.testing.assert_close(p,d(tokens,rgb),rtol=0,atol=0)
    p.log().mean().backward()
    assert all(p.grad is None for p in d.parameters())
    assert model.decoder.proj[0].weight.grad.abs().sum()>0
    assert model.detail.fusion[-1].weight.grad.abs().sum()>0


def test_edge_loss_direction_and_invalid():
    gt=torch.ones(1,1,16,16);gt[...,8:]=2
    v=torch.ones_like(gt)
    assert reliable_edge_loss(gt,gt,v)==0
    assert reliable_edge_loss(torch.ones_like(gt),gt,v)>0
    gt[...,0,:]=float("nan")
    p=torch.ones_like(gt,requires_grad=True)
    loss=reliable_edge_loss(p,gt,v);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all()


def test_toy_boundary_threshold_ignores_tiny_flat_noise():
    from scripts.joint_decoder_study import toy_boundary_f1
    gt=torch.ones(1,1,32,32)*4;gt[...,8:24,8:24]=2
    assert toy_boundary_f1(gt,gt)==1.
    assert toy_boundary_f1(torch.ones_like(gt),gt)==0.
    assert toy_boundary_f1(gt+torch.rand_like(gt)*1e-4,gt)==1.
