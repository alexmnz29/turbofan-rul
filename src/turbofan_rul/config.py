"""Paths and column definitions for the C-MAPSS dataset."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

ID_COLS = ["unit", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = ID_COLS + SETTING_COLS + SENSOR_COLS

# Piecewise-linear RUL label cap, standard in the C-MAPSS literature.
RUL_CAP = 125

# Development protocol (notebook 03). The FD001 test set is used once, at the end;
# every model before that is compared out-of-fold on the training engines.
N_FOLDS = 5
# Cut points mimic the test set: one prediction per engine at a truncated history.
MIN_HISTORY = 30  # cycles observed before a cut; the longest window planned for level 3
PLANNING_LIMIT = 150  # only engines within this many cycles of failure matter for shop planning
N_CUTS_PER_ENGINE = 10
RANDOM_SEED = 0

# Sensor decisions from notebook 01 (FD001 only: re-derive before using another subset).
FD001_RAW_SENSORS = [f"sensor_{i}" for i in (2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20, 21)]
# Trend sign differs by engine; informative only as deviation from the engine's own baseline.
FD001_DEVIATION_SENSORS = ["sensor_9", "sensor_14"]

# Window features (notebook 04). Trailing windows only; see features.py.
FEATURE_WINDOWS = (5, 10, 30)
SLOPE_WINDOWS = (10, 30)
BASELINE_CYCLES = 20  # an engine's healthy reference: mean of its first cycles

# Working model (notebook 04): LightGBM on the window features.
LGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "random_state": 0,
}
EARLY_STOPPING_ROUNDS = 100
MONITOR_ENGINES = 16  # engines held out of each training fold to monitor early stopping
