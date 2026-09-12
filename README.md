# turbofan-rul

Remaining Useful Life (RUL) prediction for aircraft turbofan engines using the NASA C-MAPSS run-to-failure dataset.

The goal is not just to minimize RMSE on a benchmark: the project frames RUL prediction as a maintenance decision problem, where predicting a failure **too late** (unplanned downtime) costs far more than predicting it **too early** (premature intervention). Models are trained and evaluated against an asymmetric cost function, following the PHM08 challenge scoring defined in Saxena et al. (2008).

## Dataset

[C-MAPSS](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (Commercial Modular Aero-Propulsion System Simulation), four subsets:

| Subset | Operating regimes | Fault modes |
|--------|-------------------|-------------|
| FD001  | 1                 | 1 (HPC degradation) |
| FD002  | 6                 | 1 (HPC degradation) |
| FD003  | 1                 | 2 (HPC + fan degradation) |
| FD004  | 6                 | 2 (HPC + fan degradation) |

Each engine starts with random initial wear and degrades until failure. Sensors are flat for most of the engine's life and only drift near the end, which motivates piecewise-linear RUL labels (capped at a maximum) and per-regime normalization.

See [data/README.md](data/README.md) for download instructions. Raw data is not committed to the repo.

## Approach

1. **EDA** — sensor trajectories per engine, constant/uninformative sensors, life distributions (Weibull fits), regime clustering.
2. **Baselines** — linear regression on last-value features, validated with engine-wise splits (never row-wise: rows from the same engine leak).
3. **Window features + gradient boosting** — rolling means, slopes and dispersion over multiple window sizes, fed to LightGBM with early stopping on a held-out engine split. This is the working model.
4. **Asymmetric cost** — quantile objectives and a custom cost function (cost of unplanned stop vs. cost of early intervention), compared against the MSE-trained model on the PHM08 score.
5. **Sequence models (optional)** — a small LSTM under the same validation protocol; kept only if it clearly beats boosting.

## Project structure

```
├── data/               # raw + processed data (gitignored, see data/README.md)
├── docs/learning/      # study notes: reliability stats → boosting → asymmetric losses
├── notebooks/          # exploratory analysis, numbered in order
├── src/turbofan_rul/   # importable package: loading, features, evaluation
└── tests/
```

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Download the C-MAPSS data (see [data/README.md](data/README.md)), then start with the notebooks:

```bash
uv run jupyter lab
```

Run tests:

```bash
uv run pytest
```

## References

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation.* PHM08.
- Heimes, F. O. (2008). *Recurrent neural networks for remaining useful life estimation.* PHM08.
- Zheng, S., et al. (2017). *Long Short-Term Memory Network for Remaining Useful Life estimation.* ICPHM.
- Li, X., Ding, Q., & Sun, J.-Q. (2018). *Remaining useful life estimation in prognostics using deep convolution neural networks.* Reliability Engineering & System Safety.

## License

MIT
