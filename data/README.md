# Data

Raw data is **not** committed to the repository.

## Download C-MAPSS

The Turbofan Engine Degradation Simulation dataset is published by the NASA Prognostics Center of Excellence:

- Landing page: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Direct mirror (NASA data portal): https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Download the C-MAPSS zip and extract it into `data/raw/`, so the layout is:

```
data/
└── raw/
    ├── train_FD001.txt
    ├── test_FD001.txt
    ├── RUL_FD001.txt
    ├── train_FD002.txt
    ├── ...
    └── RUL_FD004.txt
```

## File format

Each `train_*`/`test_*` file is whitespace-separated with 26 columns and no header:

| Columns | Content |
|---------|---------|
| 1       | engine unit number |
| 2       | time, in cycles |
| 3–5     | operational settings |
| 6–26    | sensor measurements 1–21 |

- `train_*`: each engine runs until failure (last cycle = failure).
- `test_*`: each engine's series is truncated before failure.
- `RUL_*`: true remaining cycles at the last observed cycle of each test engine.

Use `turbofan_rul.data_loading.load_subset()` to load a subset with proper column names.

`data/processed/` is for intermediate artifacts (feature tables, etc.) and is also gitignored.
