# Notebooks

Numbered in working order. Each notebook should be runnable top-to-bottom after `uv sync` and downloading the data (see `data/README.md`).

Planned sequence:

| Notebook | Content | Study level |
|----------|---------|-------------|
| `01_eda_fd001.ipynb` | Sensor trajectories of 3 to 4 engines, constant sensors, life distributions | 0 |
| `02_weibull_lifetimes.ipynb` | Lifetime distributions (Weibull, lognormal), hazard, clock-model RUL baseline, FD001 vs FD003, censoring | 1 |
| `03_linear_baseline.ipynb` | Last-value linear baseline; engine-wise vs row-wise split leakage demo | 2 |
| `04_features_and_boosting.ipynb` | Window features (short windows, differences, slopes, per-engine baseline, age) and LightGBM with early stopping; linear model on the same features as control; SHAP | 3 and 4 |
| `05_asymmetric_cost.ipynb` | Quantile and custom objectives vs MSE under the PHM08 score and the scenario's costs | 5 |
| `06_maintenance_planning.ipynb` | Threshold rule, greedy heuristic and MILP for the weekly shop plan under capacity | 6 |

Reusable logic gets promoted from notebooks into `src/turbofan_rul/` as it stabilizes. The production part (level 7) is built as a pipeline module and scripts rather than a notebook.

Budget per notebook: one Decisions table at the end, at most five reading cells, and no theory (that goes in `docs/learning/`). Every model is scored with `engine_folds`, `sample_cut_points` and `summarize`, reported as the mean over 30 cut draws with per-fold scores, and logged to MLflow with `log_model_run`.
