import pandas as pd

from turbofan_rul.data_loading import add_rul_labels


def _toy_frame() -> pd.DataFrame:
    # Two engines: unit 1 lives 3 cycles, unit 2 lives 2 cycles.
    return pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})


def test_rul_counts_down_to_zero_per_unit():
    labeled = add_rul_labels(_toy_frame(), cap=None)
    assert labeled["rul"].tolist() == [2, 1, 0, 1, 0]


def test_rul_cap_clips_early_life():
    labeled = add_rul_labels(_toy_frame(), cap=1)
    assert labeled["rul"].tolist() == [1, 1, 0, 1, 0]
