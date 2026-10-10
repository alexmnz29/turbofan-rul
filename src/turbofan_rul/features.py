"""Window features over sensor trajectories (study level 3, notebook 04).

Every feature at cycle t uses only cycles 1..t of the same engine, so a feature
computed on a full training history equals the one computed at a cut:

- windows are trailing (pandas `rolling` is trailing by default);
- everything is computed per engine with `groupby("unit")`, so no window crosses
  from one engine into the next;
- the per-engine baseline uses the engine's own first `BASELINE_CYCLES` cycles,
  which exist at any cut after that point (the protocol requires 30).

Fleet-level statistics (scalers, regime means) are not computed here; they belong
inside the training folds.
"""

import numpy as np
import pandas as pd

from turbofan_rul.config import (
    BASELINE_CYCLES,
    FD001_DEVIATION_SENSORS,
    FD001_RAW_SENSORS,
    FEATURE_WINDOWS,
    SLOPE_WINDOWS,
)


def rolling_slope(s: pd.Series, window: int) -> pd.Series:
    """Least-squares slope of the last `window` values against their position, per cycle.

    Closed form: beta = cov(x, y) / var(x) with x = 0..window-1, so the rolling
    covariance is all that is needed. Returns NaN until the window is full.
    """
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered**2).sum())
    return s.rolling(window, min_periods=window).apply(
        lambda y: float((x_centered * (y - y.mean())).sum() / denom), raw=True
    )


def build_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
    deviation_sensors: list[str] | None = None,
    windows: tuple[int, ...] = FEATURE_WINDOWS,
    slope_windows: tuple[int, ...] = SLOPE_WINDOWS,
    baseline_cycles: int = BASELINE_CYCLES,
) -> pd.DataFrame:
    """Return `df` with the window features appended, computed per engine.

    Per sensor in `sensors`: trailing mean per window, difference between the
    current reading and the longest mean, slope per slope window, rolling std over
    the longest window, and deviation of the shortest mean from the engine's
    baseline (mean of its first `baseline_cycles` cycles).
    Per sensor in `deviation_sensors` (trend sign differs by engine): absolute
    deviation from the baseline only.
    Plus `age`, the cycle number.

    Rows must be sorted by unit and cycle. Slopes and std are NaN while their
    window is incomplete; the protocol's minimum history keeps those rows out of
    evaluation, and tree models accept NaN.
    """
    sensors = sensors or FD001_RAW_SENSORS
    deviation_sensors = deviation_sensors or FD001_DEVIATION_SENSORS
    if not df.groupby("unit")["cycle"].apply(lambda c: c.is_monotonic_increasing).all():
        raise ValueError("rows must be sorted by unit and cycle")

    out = df.copy()
    grouped = out.groupby("unit", sort=False)
    longest, shortest = max(windows), min(windows)
    new: dict[str, pd.Series] = {}

    for w in windows:
        means = grouped[sensors].transform(lambda c, w=w: c.rolling(w, min_periods=1).mean())
        for s in sensors:
            new[f"{s}_mean{w}"] = means[s]
    for s in sensors:
        new[f"{s}_diff{longest}"] = out[s] - new[f"{s}_mean{longest}"]
    for w in slope_windows:
        slopes = grouped[sensors].transform(lambda c, w=w: rolling_slope(c, w))
        for s in sensors:
            new[f"{s}_slope{w}"] = slopes[s]
    stds = grouped[sensors].transform(lambda c: c.rolling(longest, min_periods=2).std())
    for s in sensors:
        new[f"{s}_std{longest}"] = stds[s]

    baseline = grouped[sensors + deviation_sensors].transform(
        lambda c: c.iloc[:baseline_cycles].mean()
    )
    for s in sensors:
        new[f"{s}_dev"] = new[f"{s}_mean{shortest}"] - baseline[s]
    short_dev = grouped[deviation_sensors].transform(
        lambda c: c.rolling(shortest, min_periods=1).mean()
    )
    for s in deviation_sensors:
        new[f"{s}_absdev"] = (short_dev[s] - baseline[s]).abs()

    new["age"] = out["cycle"].astype(float)
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Names of the columns `build_features` added to `df`."""
    suffixes = ("_mean", "_diff", "_slope", "_std", "_dev", "_absdev")
    return [c for c in df.columns if c == "age" or any(s in c for s in suffixes)]
