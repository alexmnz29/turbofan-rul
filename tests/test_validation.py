import pandas as pd

from turbofan_rul.validation import engine_folds, sample_cut_points


def _fleet(lifetimes: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [(u, c) for u, life in enumerate(lifetimes, start=1) for c in range(1, life + 1)],
        columns=["unit", "cycle"],
    )


def test_every_engine_in_exactly_one_fold_and_folds_balanced():
    df = _fleet([100 + 10 * i for i in range(20)])
    folds = engine_folds(df, n_folds=5)
    assert sorted(folds.index) == list(range(1, 21))
    assert folds.value_counts().tolist() == [4, 4, 4, 4, 4]


def test_long_lived_engines_spread_across_folds():
    df = _fleet([100 + 10 * i for i in range(20)])
    folds = engine_folds(df, n_folds=5)
    five_longest = folds.loc[[16, 17, 18, 19, 20]]
    assert sorted(five_longest) == [0, 1, 2, 3, 4]


def test_cuts_respect_history_and_planning_limit():
    df = _fleet([60, 300])
    cuts = sample_cut_points(df, n_cuts=10, min_history=30, planning_limit=150)
    assert (cuts["cycle"] >= 30).all()
    assert cuts["rul_true"].between(1, 150).all()
    lifetime = cuts["unit"].map({1: 60, 2: 300})
    assert (cuts["rul_true"] == lifetime - cuts["cycle"]).all()


def test_cut_sampling_is_reproducible():
    df = _fleet([200, 250, 300])
    a = sample_cut_points(df, n_cuts=5, seed=1)
    b = sample_cut_points(df, n_cuts=5, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("unit").size().tolist() == [5, 5, 5]
