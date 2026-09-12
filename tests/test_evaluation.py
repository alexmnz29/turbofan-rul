import numpy as np

from turbofan_rul.evaluation import phm08_score, rmse


def test_perfect_prediction_scores_zero():
    y = np.array([10.0, 50.0, 125.0])
    assert phm08_score(y, y) == 0.0
    assert rmse(y, y) == 0.0


def test_late_costs_more_than_early():
    y_true = np.array([100.0])
    early = phm08_score(y_true, y_true - 20)  # predicted 20 cycles early
    late = phm08_score(y_true, y_true + 20)  # predicted 20 cycles late
    assert late > early > 0


def test_rmse_known_value():
    assert rmse(np.array([0.0, 0.0]), np.array([3.0, -3.0])) == 3.0
