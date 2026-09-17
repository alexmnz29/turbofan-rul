# Notebooks

Numbered in working order. Each notebook should be runnable top-to-bottom after `uv sync` and downloading the data (see `data/README.md`).

Planned sequence:

| Notebook | Content | Study level |
|----------|---------|-------------|
| `01_eda_fd001.ipynb` | Sensor trajectories of 3–4 engines, constant sensors, life distributions | — |
| `02_weibull_lifetimes.ipynb` | Lifetime distributions (Weibull, lognormal), hazard, clock-model RUL baseline, FD001 vs FD003, censoring | 1 |
| `03_linear_baseline.ipynb` | Last-value linear baseline; engine-wise vs row-wise split leakage demo | 2 |
| `04_window_features.ipynb` | Rolling features across window sizes; separation of healthy vs near-failure | 3 |
| `05_gradient_boosting.ipynb` | LightGBM working model with early stopping; importances and SHAP | 4 |
| `06_asymmetric_cost.ipynb` | Quantile/custom objectives vs MSE under the PHM08 score | 5 |
| `07_lstm.ipynb` (optional) | Small LSTM under the same validation protocol | 6 |

Reusable logic gets promoted from notebooks into `src/turbofan_rul/` as it stabilizes.
