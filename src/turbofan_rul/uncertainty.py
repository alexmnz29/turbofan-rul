"""Prediction intervals from quantile models (study level 5, notebook 05).

Quantile models trained with the pinball loss are over-confident on new engines, and
separately trained quantiles can cross. Two small corrections:

- `conformal_offset`: split-conformal shift for one quantile level, computed on
  residuals from engines the model did not see;
- `sort_quantiles`: per-row sorting so that lower quantiles never exceed higher ones.
"""

import numpy as np


def conformal_offset(y_true, q_pred, alpha: float) -> float:
    """Shift to add to a fitted alpha-quantile so that P(y <= q + shift) is about alpha.

    `y_true` and `q_pred` are the outcomes and the model's alpha-quantile predictions on
    calibration data (engines held out from training). The shift is the empirical
    alpha-quantile of the residuals y - q, with the finite-sample correction of split
    conformal prediction.
    """
    residuals = np.asarray(y_true, dtype=float) - np.asarray(q_pred, dtype=float)
    n = len(residuals)
    if n == 0:
        raise ValueError("calibration set is empty")
    level = min(1.0, np.ceil((n + 1) * alpha) / n)
    return float(np.quantile(residuals, level))


def sort_quantiles(*quantile_preds) -> tuple[np.ndarray, ...]:
    """Given predictions for increasing quantile levels, sort them per row to remove crossing."""
    stacked = np.sort(np.vstack([np.asarray(q, dtype=float) for q in quantile_preds]), axis=0)
    return tuple(stacked)
