import torch
from sokkanaem.mixture_readout import AnchoredMixtureReadout, supervised_loss
from sokkanaem.detail_feature_path import DetailFeaturePath


def test_initial_mean_and_candidate_selection():
    m = AnchoredMixtureReadout()
    coarse = torch.rand(2,1,16,16)+.3
    out = m(torch.rand(2,16,8,8), coarse)
    torch.testing.assert_close(out["mean"],coarse)
    torch.testing.assert_close(out["mode"].log(),out["means"].gather(1,out["choice"]))
    assert torch.isfinite(out["mode"]).all() and (out["mode"]>0).all()


def test_density_choice_accounts_for_scale():
    m=AnchoredMixtureReadout()
    with torch.no_grad():
        m.net[-1].bias[1]=0.5
        m.net[-1].bias[2]=0.3  # first component has more mass
        m.net[-1].bias[3]=0.5  # but much lower peak density
        m.net[-1].bias[4]=-4.
    out=m(torch.zeros(1,16,8,8),torch.ones(1,1,16,16))
    assert (out["choice"]==1).all()


def test_invalid_loss_and_gradients():
    m=AnchoredMixtureReadout()
    coarse=torch.ones(2,1,16,16)
    out=m(torch.rand(2,16,8,8),coarse)
    gt=coarse*1.1; gt[...,0,:]=float("nan")
    loss,_=supervised_loss(out["mode"],gt,torch.isfinite(gt),out)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    empty,_=supervised_loss(out["mode"],gt,torch.zeros_like(gt),out)
    assert empty==0


def test_detail_feature_identity_and_gradient():
    m=DetailFeaturePath()
    rgb=torch.rand(2,3,32,32);features=torch.rand(2,16,16,16)
    out=m(rgb,features)
    assert torch.equal(out,features)
    out.square().mean().backward()
    assert torch.isfinite(m.fusion[-1].weight.grad).all()
    assert m.fusion[-1].weight.grad.abs().sum()>0
