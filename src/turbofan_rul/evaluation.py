"""Evaluation metrics: symmetric baselines and asymmetric prognostic costs."""

import numpy as np


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric scoring function from the PHM08 challenge (Saxena et al., 2008).

    Late predictions (pred > true, i.e. the engine fails before we expect it to)
    are penalized exponentially harder than early ones. Lower is better.

        d = pred - true
        score = sum( exp(-d/13) - 1 )  for d < 0   (early)
                sum( exp( d/10) - 1 )  for d >= 0  (late)
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def phm08_per_100(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM08 score rescaled to 100 predictions.

    The raw score is a sum, so it grows with the number of predictions. Rescaling puts
    protocols with different numbers of cut points on the scale of the 100-engine test set.
    """
    return 100 * phm08_score(y_true, y_pred) / len(np.asarray(y_true))


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """The scores reported for every model: RMSE, PHM08 per 100 predictions, mean error.

    Mean error is pred - true: positive means late on average, the expensive direction.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        "rmse": rmse(y_true, y_pred),
        "phm08_per_100": phm08_per_100(y_true, y_pred),
        "mean_error": float(d.mean()),
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(d**2)))


def maintenance_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_late: float,
    cost_early_per_cycle: float,
) -> float:
    """Per-prediction maintenance cost (level 5 sketch; the decision-level cost with shop
    capacity lives in level 7).

    A late prediction (pred > true) triggers one unplanned failure, a flat cost; an early
    one (pred < true) wastes the cycles between the prediction and the true failure, at a
    per-cycle cost. An exact prediction costs nothing.

    With the README scenario (failure = 10 shop visits, wasted cycle = 1/200 visit) this
    per-prediction cost is nearly minimised by predicting zero, because it omits the cost
    of the shop visit itself and the capacity limit: those enter at the decision level.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    late = d > 0
    return float(np.sum(late) * cost_late + np.sum(-d[~late]) * cost_early_per_cycle)
