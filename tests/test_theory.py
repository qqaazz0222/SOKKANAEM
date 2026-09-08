"""Numerical sanity checks and counterexamples for paper/THEORY_12.md.

Tests complement analytic proofs, and do not certify real-data accuracy.
"""
import copy

import numpy as np
import pytest
import torch

from sokkanaem import ChangeDetector, SOKKANAEM, SelectiveSSM
from sokkanaem.model import SpatialBlock, TemporalBlock
from sokkanaem.theory import (cost_speedup, diagonal_comparison, effective_activity,
    periodic_refresh_bound, pixel_cache_bound, propagate_bound)


@pytest.mark.parametrize("seed", range(5))
def test_defect_identity_and_bound(seed):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(.05, .999, (90, 7))
    q = rng.normal(size=phi.shape)
    mask = rng.integers(0, 2, size=phi.shape)
    run = diagonal_comparison(phi, q, mask, rng.normal(size=7), rng.normal(size=7))
    errors = run['gated'] - run['dense']
    np.testing.assert_allclose(errors[1:], phi*errors[:-1] + run['residuals'], atol=2e-14)
    assert np.all(run['errors'] <= run['bounds'] + 2e-14)


def test_defect_bound_sharp_scalar_case_and_constant_input_counterexample():
    # Constant pixels/tokens: no change, but the dense state still approaches equilibrium.
    run = diagonal_comparison(np.full((4, 1), .5), np.ones((4, 1)),
                              np.zeros((4, 1)), [0.], [0.])
    np.testing.assert_allclose(run['errors'], run['bounds'])
    assert run['errors'][-1] == 1.875


def test_actual_selective_ssm_shared_input_matches_affine_recurrence():
    torch.manual_seed(1201)
    model = SelectiveSSM(4, d_state=3).double().eval()
    u = torch.randn(1, 9, 4, dtype=torch.double)
    with torch.no_grad():
        x, _, dt, b, _ = model._params(u, None)
        phi = (dt[..., None] * -model.A_log.exp()).exp()[0].reshape(9, -1)
        q = ((dt*x)[..., None] * b[:, :, None, :])[0].reshape(9, -1)
        masks = torch.tensor([1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=torch.double)
        hd = torch.randn(1, 8, 3, dtype=torch.double)
        hg = hd.clone()
        comparison = diagonal_comparison(phi.numpy(), q.numpy(), masks[:, None].numpy(),
                                          hd.flatten().numpy(), hg.flatten().numpy())
        for t in range(9):
            _, hd = model.step(u[:, t], None, hd)
            _, hg = model.step(u[:, t], masks[t:t+1], hg)
            np.testing.assert_allclose(hd.flatten().numpy(), comparison['dense'][t+1], atol=1e-12)
            np.testing.assert_allclose(hg.flatten().numpy(), comparison['gated'][t+1], atol=1e-12)


@pytest.mark.parametrize("rho,period", [(0., 1), (0., 5), (.5, 1), (.5, 4), (.999, 30)])
def test_periodic_refresh_not_reset_and_geometric_cycle(rho, period):
    run = periodic_refresh_bound(rho, .2, period, 15, initial=.8)
    limit = run['limiting_after_refresh']
    expected = np.array([rho**(period*j)*.8 + (1-rho**(period*j))*limit for j in range(16)])
    np.testing.assert_allclose(run['after_refresh'], expected, atol=3e-12)
    if rho > 0:
        assert run['after_refresh'][1] > 0  # not a state reset


def test_refresh_with_distinct_inputs_has_additive_defect():
    # Negative A does not make a stacked input/state feedback map contractive.
    # F(h)=.5h + u(h), u(h)=h: the complete map expands by 1.5.
    dense, sparse = 1.0, 2.0
    assert abs((.5*sparse+sparse)-(.5*dense+dense)) == 1.5*abs(sparse-dense)


def test_actual_model_keyframe_uses_carried_state_not_zero_or_dense_history():
    torch.manual_seed(1202)
    m = SOKKANAEM(dim=8, depth=2, d_state=2, spatial_cache=True,
                 temporal_cache=True, keyframe_every=3).double().eval()
    frame = torch.rand(1, 3, 32, 32, dtype=torch.double)
    with torch.no_grad():
        _, state, _ = m.step(frame)
        for _ in range(2):
            _, state, _ = m.step(frame, state)
        old = copy.deepcopy(state)
        output, refreshed, info = m.step(frame, state)
        all_on = frame.new_ones(1, 4)
        expected, hs = m._step_core(frame, all_on, old['hs'])
        _, zero_hs = m._step_core(frame, all_on, None)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert info['active_ratio'] == 1.
        for a, b in zip(refreshed['hs'], hs):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert any(not torch.allclose(a, b) for a, b in zip(hs, zero_hs))


@pytest.mark.parametrize("period", [1, 2, 7, 30])
def test_detector_age_and_finite_horizon_refresh_count(period):
    det = ChangeDetector(patch_size=2, keyframe_every=period)
    frame = torch.zeros(1, 3, 4, 4)
    st, age, keyframes, horizon = None, 0, 0, 73
    for t in range(horizon):
        mask, st = det(frame, st)
        if bool(mask.all()):
            keyframes += 1
            age = 0
        else:
            age += 1
        assert age <= period-1
    assert keyframes == (horizon+period-1)//period


def test_cache_bound_requires_drift_sum_not_only_latest_frame():
    pixels, age, delta, w, lip = 12, 7, .01, 2., 3.
    drift = np.linalg.norm(np.full(pixels, age*delta))
    bound = pixel_cache_bound(age, pixels, delta**2, w, lip)
    assert bound == pytest.approx(lip*w*drift)
    assert bound > pixel_cache_bound(1, pixels, delta**2, w, lip)


def test_spatial_gather_has_context_defect_even_for_identical_tokens():
    torch.manual_seed(1203)
    block = SpatialBlock(4, 2).double().eval()
    u = torch.randn(1, 4, 4, dtype=torch.double)
    mask = torch.tensor([[0., 1., 0., 0.]], dtype=torch.double)
    with torch.no_grad():
        dense = block(u, (2, 2))
        sparse, _ = block.forward_cached(u, mask, dense, (2, 2))
    assert not torch.allclose(sparse[:, 1], dense[:, 1], rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(sparse[:, [0, 2, 3]], dense[:, [0, 2, 3]], rtol=0, atol=0)


def test_temporal_cache_holds_output_but_uncached_readout_changes():
    torch.manual_seed(1204)
    block = TemporalBlock(4, 2).double().eval()
    u, v = torch.randn(1, 3, 4, dtype=torch.double), torch.randn(1, 3, 4, dtype=torch.double)
    with torch.no_grad():
        cache, h = block.step(u, u.new_ones(1, 3), None)
        held, h2, _ = block.step_cached(v, v.new_zeros(1, 3), h, cache)
        fresh, h3 = block.step(v, v.new_zeros(1, 3), h)
    assert torch.equal(h, h2) and torch.equal(h, h3)
    assert torch.equal(held, cache) and not torch.allclose(held, fresh)


def test_amdahl_overhead_and_overlap_accounting():
    assert cost_speedup(.7, 0.) == pytest.approx(1/.7)
    assert cost_speedup(.7, .2, .3) < 1.
    assert cost_speedup(.7, 1/30) < cost_speedup(.7, 0.)
    a = effective_activity([.1, .2, .3, .4], [True, False, False, False], [True, False, True, False])
    np.testing.assert_array_equal(a, [1., .2, 1., .4])


def test_general_noncontractive_bound_and_reject_invalid_assumptions():
    np.testing.assert_allclose(propagate_bound([2., .5], [.1, .2], 1.), [1., 2.1, 1.25])
    with pytest.raises(ValueError):
        periodic_refresh_bound(1., .2, 3, 4)
    with pytest.raises(ValueError):
        propagate_bound([.5], [-1.])
    with pytest.raises(ValueError):
        diagonal_comparison([[.5]], [[1.]], [[.5]], [0.], [0.])
    with pytest.raises(ValueError):
        cost_speedup(.5, 1.1)


def test_zoh_input_approximation_bound():
    for lam in (.01, .5, 10., 100.):
        for delta in (0., 1e-6, .1, 1., 10.):
            difference = delta + np.expm1(-lam*delta)/lam
            assert difference >= -1e-15
            assert difference <= lam*delta**2/2 + 1e-14


def test_fractional_mask_is_not_update_copy_interpolation():
    assert np.exp(-.5) != pytest.approx(.5*np.exp(-1.)+.5)


def test_absrel_difference_bound_requires_same_unaligned_predictions():
    rng = np.random.default_rng(1210)
    gt = rng.uniform(.01, 10., 100)
    a, b = rng.uniform(.01, 10., (2, 100))
    difference = abs(np.mean(np.abs(a-gt)/gt)-np.mean(np.abs(b-gt)/gt))
    assert difference <= np.mean(np.abs(a-b)/gt)
    assert np.mean(np.abs(a-b)/gt) <= np.mean(np.abs(a-b))/gt.min()


def test_joint_cache_cost_interval_arithmetic():
    fixed, active, overhead, target = .3, .1, .01, 1.5
    coefficient, cache_tolerance = .1, .5
    denominator = 1/target-fixed-(1-fixed)*active-overhead
    kmin = max(1, int(np.ceil((1-fixed)*(1-active)/denominator)))
    kmax = 1+int(np.floor(cache_tolerance/coefficient))
    for k in range(1, 20):
        meets_cost = cost_speedup(fixed, active+(1-active)/k, overhead) >= target
        meets_cache = coefficient*(k-1) <= cache_tolerance
        assert (meets_cost and meets_cache) == (kmin <= k <= kmax)
