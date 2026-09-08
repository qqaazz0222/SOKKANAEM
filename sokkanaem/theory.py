"""Task 12: executable algebra, not an inference policy or a learned certificate.

All arrays use float64. Bounds assume the supplied constants are valid; sampled
checks do not establish global Lipschitz constants for the trained model.
"""
import math

import numpy as np


def propagate_bound(lipschitz, defects, initial=0.0):
    """E[t] <= L[t] E[t-1] + d[t], including E[0] before any update."""
    ls, ds = np.asarray(lipschitz, dtype=float), np.asarray(defects, dtype=float)
    if ls.ndim != 1 or ls.shape != ds.shape:
        raise ValueError("one-dimensional equal-length sequences required")
    if not np.isfinite(initial) or initial < 0 or not np.isfinite(ls).all() \
            or not np.isfinite(ds).all() or np.any(ls < 0) or np.any(ds < 0):
        raise ValueError("finite nonnegative bounds required")
    out = [float(initial)]
    for lip, defect in zip(ls, ds):
        out.append(lip * out[-1] + defect)
    return np.asarray(out)


def diagonal_comparison(phi, drive, masks, dense_initial, gated_initial):
    """Shared-input diagonal affine recurrence and a posteriori defect bound.

    phi/drive/masks: (time, flattened state); binary masks may be broadcast
    from (time, 1). Norm is Euclidean. No token/parameter feedback is assumed.
    """
    phi, drive = np.asarray(phi, float), np.asarray(drive, float)
    if phi.ndim != 2 or phi.shape != drive.shape or not np.isfinite(phi).all() \
            or not np.isfinite(drive).all() or np.any(phi < 0) or np.any(phi > 1):
        raise ValueError("finite matching (time, state) arrays, 0 <= phi <= 1 required")
    masks = np.broadcast_to(np.asarray(masks, float), phi.shape)
    if not np.isin(masks, [0., 1.]).all():
        raise ValueError("binary masks required")
    d, g = np.asarray(dense_initial, float), np.asarray(gated_initial, float)
    if d.shape != (phi.shape[1],) or g.shape != d.shape \
            or not np.isfinite(d).all() or not np.isfinite(g).all():
        raise ValueError("finite state vectors required")
    dense, gated, residuals, errors = [d.copy()], [g.copy()], [], [np.linalg.norm(g-d)]
    for p, q, m in zip(phi, drive, masks):
        residuals.append((1-m) * ((1-p)*g-q))
        d = p*d + q
        g = m*(p*g+q) + (1-m)*g
        dense.append(d.copy())
        gated.append(g.copy())
        errors.append(np.linalg.norm(g-d))
    residuals = np.asarray(residuals).reshape(phi.shape)
    defects = np.linalg.norm(residuals, axis=1)
    bounds = propagate_bound(phi.max(axis=1), defects, errors[0])
    return dict(dense=np.asarray(dense), gated=np.asarray(gated),
                residuals=residuals, defects=defects, errors=np.asarray(errors), bounds=bounds)


def periodic_refresh_bound(rho, epsilon, period, cycles, initial=0.0):
    """After a refresh: K-1 bounded defects, then a zero-defect dense update.

    Common-input contraction rho is assumed at EVERY step. Refresh does not
    reset the state. This is not a whole-network certificate.
    """
    if not 0 <= rho < 1 or not math.isfinite(epsilon) or epsilon < 0 \
            or not isinstance(period, int) or period < 1 \
            or not isinstance(cycles, int) or cycles < 1:
        raise ValueError("rho in [0,1), epsilon >= 0, positive integer K/cycles required")
    defect_cycle = [epsilon] * (period-1) + [0.0]
    path = propagate_bound([rho] * (period*cycles), defect_cycle * cycles, initial)
    injection = rho * epsilon * sum(rho**j for j in range(period-1))
    return dict(path=path, after_refresh=path[::period],
                limiting_after_refresh=injection / (1-rho**period))


def pixel_cache_bound(age, patch_values, tau, embedding_norm, readout_lipschitz):
    """Conditional first-block cache bound L_R ||W|| age sqrt(C p^2 tau)."""
    vals = [age, patch_values, tau, embedding_norm, readout_lipschitz]
    if not all(math.isfinite(v) and v >= 0 for v in vals) or patch_values <= 0:
        raise ValueError("nonnegative finite inputs and positive patch size required")
    return readout_lipschitz * embedding_norm * age * math.sqrt(patch_values*tau)


def cost_speedup(fixed_fraction, active_fraction, overhead_fraction=0.0):
    """Ideal additive cost: S=1/[f+(1-f)a+omega], dense reference normalized to 1."""
    if not 0 <= fixed_fraction <= 1 or not 0 <= active_fraction <= 1 \
            or not math.isfinite(overhead_fraction) or overhead_fraction < 0:
        raise ValueError("fractions f,a in [0,1] and finite nonnegative overhead required")
    denom = fixed_fraction + (1-fixed_fraction)*active_fraction + overhead_fraction
    return math.inf if denom == 0 else 1/denom


def effective_activity(raw_activity, keyframes, fallbacks):
    """Union of keyframe/fallback events; never double-count overlapping events."""
    raw = np.asarray(raw_activity, float)
    keys, falls = np.asarray(keyframes, bool), np.asarray(fallbacks, bool)
    if raw.ndim != 1 or raw.size == 0 or raw.shape != keys.shape or raw.shape != falls.shape \
            or not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1):
        raise ValueError("matching nonempty activity/event sequences required")
    return np.where(keys | falls, 1.0, raw)
