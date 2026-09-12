"""Window-based feature engineering over sensor trajectories.

To be built during level 3 of the study plan: rolling means, slopes and
dispersion per sensor over several window sizes, normalized per operating
regime for FD002/FD004.
"""

import pandas as pd

from turbofan_rul.config import SENSOR_COLS


def rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (5, 15, 30),
    sensors: list[str] | None = None,
) -> pd.DataFrame:
    """Rolling mean, std and slope per sensor and window size, computed per unit.

    Placeholder implementation: mean and std only. Slopes (linear trend per
    window) come with the level 3 exercise.
    """
    sensors = sensors or SENSOR_COLS
    out = df.copy()
    grouped = out.groupby("unit")[sensors]
    for w in windows:
        rolled = grouped.rolling(window=w, min_periods=1)
        out[[f"{s}_mean_{w}" for s in sensors]] = rolled.mean().reset_index(level=0, drop=True)
        out[[f"{s}_std_{w}" for s in sensors]] = rolled.std().reset_index(level=0, drop=True)
    return out
