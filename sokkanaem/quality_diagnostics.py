"""Separate development diagnostics; does not redefine frozen paper metrics."""
import torch
import torch.nn.functional as F
from sokkanaem.sharpness import boundary_band, grad_mag


def region_diagnostics(pred,gt,valid):
    keep=valid.bool() & torch.isfinite(gt) & (gt>0) & (gt<150)
    safe=torch.nan_to_num(gt,nan=0.,posinf=0.,neginf=0.).clamp_min(1e-6)
    error=(pred-safe).abs()/safe
    band=boundary_band(safe,keep)
    regions={"edge_absrel":band,"nonedge_absrel":keep & ~band,
             "near_absrel":keep & (safe<2),"mid_absrel":keep & (safe>=2) & (safe<5),
             "far_absrel":keep & (safe>=5)}
    out={key:float(error[mask].mean()) if mask.any() else None for key,mask in regions.items()}
    # Diagnostic only: fixed physical relative-depth discontinuity, with an
    # explicitly reported tolerance sweep (not a thin-object instance metric).
    gp,defined=grad_mag(pred.clamp_min(1e-6).log(),keep)
    gg,_=grad_mag(safe.log(),keep)
    interior=F.avg_pool2d(keep.float(),3,1,1)>.999
    bg=(gg>.02)&defined.bool()&interior
    bp=(gp>.02)&defined.bool()&interior
    for tol in (1,2,4):
        grow=lambda mask:F.max_pool2d(mask.float(),2*tol+1,1,tol)>0
        out[f"edge_recall_tol{tol}"]=float((bg&grow(bp)).sum()/bg.sum().clamp_min(1)) if bg.any() else None
        out[f"edge_precision_tol{tol}"]=float((bp&grow(bg)).sum()/bp.sum().clamp_min(1)) if bp.any() else None
    out["mean_log_scale_bias"]=float((pred.clamp_min(1e-6).log()-safe.log())[keep].mean())
    out["valid_pixels"]=int(keep.sum())
    return out


def clip_log_scale_correction(pred,gt,valid):
    """Mean log(gt/pred) on valid pixels. Caller must fit ONLY training data."""
    keep=valid.bool() & torch.isfinite(gt) & (gt>0) & (gt<150)
    if not keep.any():raise ValueError("No valid scale-fit pixels")
    return float((gt[keep].log()-pred[keep].clamp_min(1e-6).log()).mean())
