import numpy as np

from turbofan_rul.baselines import WeibullClock
from turbofan_rul.evaluation import summarize


def _lifetimes():
    rng = np.random.default_rng(0)
    return 125 + 90 * rng.weibull(1.8, size=500)


def test_clock_at_age_zero_predicts_median_lifetime():
    life = _lifetimes()
    clock = WeibullClock().fit(life)
    assert abs(clock.predict([0])[0] - np.median(life)) < 10


def test_clock_residual_life_shrinks_with_age_under_wear_out():
    clock = WeibullClock().fit(_lifetimes())
    pred = clock.predict([50, 150, 250])
    assert (pred > 0).all() and pred[0] > pred[1] > pred[2]


def test_summarize_signs_late_as_positive():
    scores = summarize(np.array([50.0, 50.0]), np.array([60.0, 70.0]))
    assert scores["mean_error"] == 15.0 and scores["rmse"] > 0
