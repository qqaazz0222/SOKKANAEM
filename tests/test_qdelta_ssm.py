import torch
from sokkanaem.qdelta_ssm import QDeltaSSM


def state(m):
    return m.refresh({"features":[torch.rand(1,5,8) for _ in range(2)],
                      "pooled":torch.rand(1,8),"grid":(2,2),"size":(28,28)})


def test_inactive_copy_and_zero_initial_readout():
    m=QDeltaSSM(channels=8,stages=2,width=8);s=state(m);rgb=torch.rand(1,3,28,28)
    z,info=m.update(rgb,torch.zeros(1,4),s)
    assert z is s and info["updated_patches"]==0
    new,info=m.update(rgb,torch.tensor([[1.,0,0,0]]),s)
    assert info["updated_patches"]==1
    assert torch.equal(new["h"][1:],s["h"][1:])
    assert not torch.equal(new["h"][0],s["h"][0])
    for a,b in zip(new["features"],s["features"]):assert torch.equal(a,b)
    assert torch.equal(new["pooled"],s["pooled"])


def test_nonzero_readout_only_changes_active_patch_tokens():
    m=QDeltaSSM(channels=8,stages=2,width=8);s=state(m)
    with torch.no_grad():m.readout.weight.normal_(0,.1)
    out,_=m.update(torch.rand(1,3,28,28),torch.tensor([[1.,0,0,0]]),s)
    for a,b in zip(out["features"],s["features"]):
        assert torch.equal(a[:,2:],b[:,2:])
    loss=sum(a.square().mean() for a in out["features"]);loss.backward()
    assert torch.isfinite(m.readout.weight.grad).all()
