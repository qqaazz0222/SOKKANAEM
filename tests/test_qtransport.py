import torch
from sokkanaem.qtransport import candidate_bank, candidate_target, TransportRefiner, OFFSETS


def test_candidate_direction_and_replicated_boundaries():
    previous=torch.arange(81).float().reshape(1,1,9,9)+1
    coarse=previous+100
    bank=candidate_bank(coarse,previous)
    assert bank.shape==(1,50,9,9) and torch.equal(bank[:,:1],coarse)
    idx=1+OFFSETS.index(0)*7+OFFSETS.index(1)
    torch.testing.assert_close(bank[:,idx,:,2:7],previous[:,0,:,3:8])
    assert torch.equal(bank[:,idx,:,-1],previous[:,0,:,-1])


def test_target_prefers_coarse_for_ties_and_finds_shift():
    previous=torch.arange(81).float().reshape(1,1,9,9)+1
    bank=candidate_bank(previous,previous)
    assert candidate_target(bank,previous).sum()==0
    coarse=torch.full_like(previous,1000)
    bank=candidate_bank(coarse,previous)
    labels=candidate_target(bank,previous)
    selected=bank.gather(1,labels[:,None])
    torch.testing.assert_close(selected,previous)


def test_hard_initial_identity_and_training_gradient():
    m=TransportRefiner();current=torch.rand(1,3,16,16);previous=torch.rand_like(current)
    coarse=torch.rand(1,1,16,16)+1;pd=coarse+1
    out,logits=m(current,previous,coarse,pd)
    torch.testing.assert_close(out,coarse,rtol=0,atol=0)
    out.mean().backward()
    assert torch.isfinite(m.fuse[-1].weight.grad).all() and m.fuse[-1].weight.grad.abs().sum()>0
    m.eval()
    with torch.no_grad():
        m.fuse[-1].bias.zero_()
        m.fuse[-1].bias[1+OFFSETS.index(0)*7+OFFSETS.index(0)]=10
    out,_=m(current,previous,coarse,pd)
    torch.testing.assert_close(out,pd,rtol=0,atol=0)
    m.selection="disabled"
    out,_=m(current,previous,coarse,pd)
    torch.testing.assert_close(out,coarse,rtol=0,atol=0)
