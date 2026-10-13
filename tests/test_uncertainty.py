import numpy as np

from turbofan_rul.uncertainty import conformal_offset, sort_quantiles


def test_conformal_offset_restores_nominal_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(50, 10, 5000)
    q10_too_high = np.full_like(y, 45.0)  # true 0.1 quantile is about 37
    shift = conformal_offset(y, q10_too_high, 0.1)
    assert shift < 0
    assert abs((y <= q10_too_high + shift).mean() - 0.1) < 0.02


def test_sort_quantiles_removes_crossing():
    lo, mid, hi = sort_quantiles([10.0, 60.0], [20.0, 50.0], [30.0, 40.0])
    assert (lo <= mid).all() and (mid <= hi).all()
    assert list(hi) == [30.0, 60.0]
