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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(d**2)))


def maintenance_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_late: float,
    cost_early_per_cycle: float,
) -> float:
    """Business-flavored cost, to be defined properly in level 5.

    Sketch: a late prediction triggers one unplanned stop (flat cost);
    an early one wastes the remaining useful cycles (per-cycle cost).
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    late = d >= 0
    return float(np.sum(late) * cost_late + np.sum(-d[~late]) * cost_early_per_cycle)
