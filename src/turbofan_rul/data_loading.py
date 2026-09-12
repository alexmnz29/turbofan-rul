"""Loading and labeling of C-MAPSS subsets."""

import pandas as pd

from turbofan_rul.config import ALL_COLS, RAW_DIR, RUL_CAP, SUBSETS


def load_subset(subset: str, split: str = "train") -> pd.DataFrame:
    """Load one C-MAPSS subset (e.g. "FD001") with named columns.

    split: "train" (run-to-failure) or "test" (truncated before failure).
    """
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    path = RAW_DIR / f"{split}_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for download instructions.")
    return pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)


def load_test_rul(subset: str) -> pd.Series:
    """True RUL at the last observed cycle of each test engine, indexed by unit."""
    path = RAW_DIR / f"RUL_{subset}.txt"
    rul = pd.read_csv(path, header=None).squeeze("columns")
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="unit")
    return rul.rename("rul")


def add_rul_labels(df: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """Add a `rul` column to a training frame: cycles remaining until that unit's failure.

    With `cap`, labels follow the standard piecewise-linear scheme: constant at `cap`
    early in life (sensors carry no degradation signal yet), then linearly decreasing.
    """
    out = df.copy()
    out["rul"] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out
