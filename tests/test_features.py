import numpy as np
import pandas as pd
import pytest

from turbofan_rul.features import build_features, feature_columns, rolling_slope


def _engine(unit: int, n: int, slope: float, offset: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cycle = np.arange(1, n + 1)
    return pd.DataFrame({
        "unit": unit, "cycle": cycle,
        "s_a": offset + slope * cycle + rng.normal(0, 0.01, n),
        "s_b": 10.0 - slope * cycle + rng.normal(0, 0.01, n),
        "s_c": offset + rng.normal(0, 0.01, n),
    })


def _fleet() -> pd.DataFrame:
    return pd.concat([_engine(1, 80, 0.05, 100.0, 0), _engine(2, 60, 0.02, 103.0, 1)], ignore_index=True)


def test_rolling_slope_recovers_a_linear_trend():
    s = pd.Series(3.0 + 0.5 * np.arange(50))
    slope = rolling_slope(s, 10)
    assert slope.iloc[:9].isna().all()
    assert np.allclose(slope.iloc[9:], 0.5)


def test_features_only_use_the_past():
    # A feature at cycle t must not change when later cycles are removed.
    full = build_features(_fleet(), sensors=["s_a", "s_b"], deviation_sensors=["s_c"])
    truncated = build_features(_fleet().query("cycle <= 40"), sensors=["s_a", "s_b"], deviation_sensors=["s_c"])
    cols = feature_columns(full)
    a = full[full["cycle"] <= 40].set_index(["unit", "cycle"])[cols]
    b = truncated.set_index(["unit", "cycle"])[cols]
    pd.testing.assert_frame_equal(a, b)


def test_windows_do_not_cross_engines():
    feats = build_features(_fleet(), sensors=["s_a"], deviation_sensors=["s_c"])
    first_of_unit_2 = feats[(feats["unit"] == 2) & (feats["cycle"] == 1)].iloc[0]
    # A 5-cycle mean at cycle 1 is the reading itself, not a blend with engine 1's last cycles.
    assert first_of_unit_2["s_a_mean5"] == pytest.approx(first_of_unit_2["s_a"])
    assert np.isnan(first_of_unit_2["s_a_slope10"])


def test_deviation_removes_the_engine_offset():
    feats = build_features(_fleet(), sensors=["s_a"], deviation_sensors=["s_c"])
    at_cycle_1 = feats[feats["cycle"] == 1].set_index("unit")
    # Engines start 3 units apart in s_a but both start near zero deviation.
    assert abs(at_cycle_1.loc[1, "s_a"] - at_cycle_1.loc[2, "s_a"]) > 2
    assert abs(at_cycle_1.loc[1, "s_a_dev"]) < 1 and abs(at_cycle_1.loc[2, "s_a_dev"]) < 1


def test_unsorted_input_is_rejected():
    with pytest.raises(ValueError):
        build_features(_fleet().iloc[::-1], sensors=["s_a"], deviation_sensors=["s_c"])
