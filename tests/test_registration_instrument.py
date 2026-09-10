import numpy as np
import pytest
from scripts.check_registration_instrument import sweep, translated


@pytest.mark.parametrize('dy,dx', [(0, 0), (2, -3), (-3, 3)])
def test_known_shift(dy, dx):
    a = np.random.default_rng(21).normal(size=(64, 80))
    r = sweep(a, translated(a, dy, dx), radius=4, stride=2)
    assert r['unique_peak']
    assert (r['best']['dy'], r['best']['dx']) == (dy, dx)
    # Padding may remove extrema: fixed marginal bin edges can differ between
    # images, so perfect spatial correspondence need not have quantized NMI=1.
    assert r['best']['score'] > .7


def test_flat_is_not_localizable():
    r = sweep(np.ones((64, 80)), np.ones((64, 80)), radius=4)
    assert not r['unique_peak'] and r['tie_count'] == 81


def test_translation_has_no_border_wrap():
    a = np.arange(48).reshape(6, 8)
    b = translated(a, 1, -2)
    assert np.isnan(b[0]).all() and np.isnan(b[:, -2:]).all()
    np.testing.assert_array_equal(b[1:, :-2], a[:-1, 2:])


def test_no_valid_support_is_rejected():
    with pytest.raises(ValueError, match='Insufficient'):
        sweep(np.full((64, 80), np.nan), np.ones((64, 80)))
