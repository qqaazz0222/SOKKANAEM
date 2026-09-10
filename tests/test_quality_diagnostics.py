import torch
from sokkanaem.quality_diagnostics import region_diagnostics,clip_log_scale_correction


def test_training_scale_and_regions():
    gt=torch.ones(1,1,32,32);gt[...,16:]=3
    pred=gt*2;valid=torch.ones_like(gt)
    correction=clip_log_scale_correction(pred,gt,valid)
    torch.testing.assert_close(torch.tensor(correction).exp(),torch.tensor(.5))
    r=region_diagnostics(pred,gt,valid)
    assert abs(r["edge_absrel"]-1)<1e-6
    assert r["edge_recall_tol1"]==1
    assert r["far_absrel"] is None


def test_recall_tolerance_monotone():
    gt=torch.ones(1,1,32,32);gt[...,16:]=3
    pred=torch.ones_like(gt);pred[...,19:]=3
    r=region_diagnostics(pred,gt,torch.ones_like(gt))
    assert r["edge_recall_tol1"]<=r["edge_recall_tol2"]<=r["edge_recall_tol4"]
    assert r["edge_recall_tol1"]==0 and r["edge_recall_tol4"]==1


def test_calibration_conversion_positive_and_bounded():
    from scripts.calibrate_pretrained_resolution import predict
    calib=torch.nn.Linear(3,2)
    dn=torch.linspace(-4,4,16).reshape(1,1,4,4)
    pred=predict(calib,dn,torch.zeros(1,3))
    assert torch.isfinite(pred).all() and pred.min()>=.3 and pred.max()<=150
    pred.log().mean().backward()
    assert torch.isfinite(calib.weight.grad).all()
