"""Development protocol: engine-wise folds and cut points that mimic the test set.

Every model from notebook 03 onward is compared with the same folds and the same cut
points, so their scores are comparable. The FD001 test set is not used here.
"""

import numpy as np
import pandas as pd

from turbofan_rul.config import (
    MIN_HISTORY,
    N_CUTS_PER_ENGINE,
    N_FOLDS,
    PLANNING_LIMIT,
    RANDOM_SEED,
)


def engine_folds(df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = RANDOM_SEED) -> pd.Series:
    """Assign every engine to one fold, with lifetimes spread evenly across folds.

    Engines are sorted by lifetime and taken in blocks of `n_folds`; inside each block the
    fold labels are shuffled. Every fold therefore gets short-lived, typical and long-lived
    engines, instead of leaving the long right tail to chance.

    Returns a Series indexed by unit with the fold number (0 to n_folds - 1).
    """
    lifetimes = df.groupby("unit")["cycle"].max().sort_values(kind="stable")
    rng = np.random.default_rng(seed)
    labels = np.empty(len(lifetimes), dtype=int)
    for start in range(0, len(lifetimes), n_folds):
        block = np.arange(start, min(start + n_folds, len(lifetimes)))
        labels[block] = rng.permutation(n_folds)[: len(block)]
    return pd.Series(labels, index=lifetimes.index, name="fold").sort_index()


def sample_cut_points(
    df: pd.DataFrame,
    n_cuts: int = N_CUTS_PER_ENGINE,
    min_history: int = MIN_HISTORY,
    planning_limit: int = PLANNING_LIMIT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Sample cut points on training engines, where a prediction is evaluated.

    A cut at cycle c means the model sees cycles 1..c and predicts the remaining life,
    as for a test engine. A cycle is eligible when at least `min_history` cycles have been
    observed and the true remaining life is between 1 and `planning_limit`. Up to
    `n_cuts` eligible cycles are drawn per engine, without replacement.

    Returns one row per cut with columns unit, cycle and rul_true (uncapped).
    """
    lifetime = df.groupby("unit")["cycle"].transform("max")
    rows = df[["unit", "cycle"]].assign(rul_true=lifetime - df["cycle"])
    eligible = rows[
        (rows["cycle"] >= min_history)
        & (rows["rul_true"] >= 1)
        & (rows["rul_true"] <= planning_limit)
    ]
    rng = np.random.default_rng(seed)
    picked = [
        g.iloc[np.sort(rng.choice(len(g), size=min(n_cuts, len(g)), replace=False))]
        for _, g in eligible.groupby("unit")
    ]
    return pd.concat(picked, ignore_index=True)
