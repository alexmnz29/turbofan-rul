# turbofan-rul

[![CI](https://github.com/alexmnz29/turbofan-rul/actions/workflows/ci.yml/badge.svg)](https://github.com/alexmnz29/turbofan-rul/actions/workflows/ci.yml)

Predictive maintenance for a fleet of aircraft turbofan engines, built on the NASA C-MAPSS run-to-failure dataset: from sensor data to a weekly maintenance plan, run as a batch pipeline.

Most C-MAPSS projects stop at a Remaining Useful Life (RUL) prediction and an RMSE. This one treats RUL as an input to a decision. A late prediction means an unplanned failure, an early one wastes engine life, and the shop can only take so many engines per week. The project predicts RUL with its uncertainty, turns it into a maintenance plan under those constraints, and measures the result in maintenance cost over a simulated period of operation.

## Scenario

The project answers a concrete operational problem. The numbers are illustrative assumptions, not sourced from an operator, and the cost and capacity parameters are varied in a sensitivity analysis.

- **Fleet.** 100 engines of one type, operating in a single flight condition (C-MAPSS FD001). Each engine flies one cycle per day and reports one reading of 21 sensors per cycle.
- **Planning.** Every Monday the maintenance planning team needs the list of engines to send to the shop that week.
- **Capacity.** The shop takes at most 5 engines per week.
- **Costs**, in units of one planned shop visit:
  - An unplanned failure costs 10 (in-service shutdown, aircraft on ground, secondary damage).
  - Removing an engine early throws away its remaining cycles, valued at 1/200 per cycle, since a shop visit buys roughly 200 cycles of life.
- **Change.** Partway through, a second fleet enters service with an additional fault mode (C-MAPSS FD003). The system has to notice that the new engines do not behave like the ones it was trained on.

**Success measure:** total maintenance cost over the simulated period, with the number of unplanned failures and the cycles wasted by early removals, compared against two reference policies: the age-only "clock" model and a model trained with plain MSE.

## From requirements to components

Every component exists because the scenario needs it.

| Requirement | Component |
|---|---|
| Estimate how long each engine has left, with its uncertainty | RUL model with quantile outputs (gradient boosting on window features) |
| Decide which engines go to the shop given 5 slots per week | Mixed-integer optimization of the weekly shop schedule |
| A new plan every week without manual steps | Batch pipeline: ingest, features, scoring, planning |
| Know which model version produced each plan | Experiment tracking and model registry (MLflow) |
| A code change must not silently break the pipeline | Continuous integration with unit tests and a smoke run |
| Detect when the fleet no longer looks like the training data | Drift monitoring on inputs and predictions, with a retraining trigger |
| Show that the system actually lowers cost | Replay of a held-out fleet week by week, with realized cost |

## Plan and status

### Part 1: Prediction

| # | Step | Output | Status |
|---|---|---|---|
| 1 | EDA: sensor screening, RUL label design | Sensor list, RUL cap of 125 | Done ([notebook 01](notebooks/01_eda_fd001.ipynb)) |
| 2 | Lifetime distributions and the clock baseline | Weibull3 fit, age-only RUL baseline | Done ([notebook 02](notebooks/02_weibull_lifetimes.ipynb)) |
| 3 | Linear baseline and the validation protocol | Engine-wise validation, leakage demo, first tracked runs, CI | Done ([notebook 03](notebooks/03_linear_baseline.ipynb)) |
| 4 | Window features and gradient boosting (one notebook) | LightGBM working model on window features, linear model on the same features as control, SHAP | Next |
| 5 | Asymmetric cost and quantile outputs | RUL quantiles, comparison with MSE under the scenario's costs | Planned |

Sequence models (LSTM) were dropped from the plan: the project's value is in the decision and production layers, and the time goes there.

### Part 2: Decision

| # | Step | Output | Status |
|---|---|---|---|
| 6 | Weekly maintenance planning | Per-engine threshold rule, greedy heuristic and MILP under shop capacity, compared on cost | Planned |

### Part 3: Production

| # | Step | Output | Status |
|---|---|---|---|
| 7 | Batch pipeline and simulation | One-command weekly run (ingest, score, plan), model registry, replay of a held-out fleet with realized cost, drift incident with FD003 and retraining | Planned |

A design decisions document will record the main choices and the alternatives left out.

Working rules, set after an audit of the first three notebooks: every notebook ends in one Decisions table and carries at most five reading cells, theory lives in the study notes, every model is reported as the mean over 30 cut draws with per-fold scores, and a difference between models counts only if it clearly exceeds the fold-to-fold spread (about 2 cycles of RMSE).

## Results so far

**Development protocol** (notebook 03). Five engine-wise folds on the 100 FD001 training engines, stratified by lifetime. Each validation engine is scored at 10 cut points with at least 30 cycles observed and 1 to 150 cycles of true remaining life, against the true, uncapped RUL. Scores are averaged over 30 draws of cut points:

| Model | RMSE | PHM08 per 100 predictions |
|---|---|---|
| Clock, capped at 125 | 32.0 ± 0.5 | 7 359 ± 989 |
| Linear regression, 12 sensors | 25.0 ± 0.4 | 1 709 ± 116 |

The linear model beats the clock on every draw and in every fold (gap 4 to 9 cycles across folds), but is systematically late between 25 and 75 cycles from failure, and the clock is still better on engines with typical lifetimes. A row-wise split would have inflated flexible models by 12 to 20 cycles of RMSE and reversed the ranking. Development scores compare with the true, uncapped remaining life, so absolute RMSE is higher than in the literature, which scores against capped labels; the final test-set table will report both.

**Test set reference** (notebook 02). Age-only clock model (conditional median residual life under a three-parameter Weibull), FD001 test set:

| Model | RMSE | PHM08 score |
|---|---|---|
| Naive clock (mean lifetime − age) | 40.2 | 25 528 |
| Clock, Weibull3 conditional median | 35.9 | 6 820 |

This is the reference for the final results table. The FD001 test set is the final exam: it is used once, at the end, for the results table in the README. During development, models are compared with engine-wise cross-validation on the training engines, using out-of-fold RMSE and PHM08, and the clock is recomputed under the same protocol. PHM08 is the asymmetric score from Saxena et al. (2008): late predictions are penalized more than early ones, and large errors grow exponentially.

## Out of scope

- **Online REST API.** Planning happens weekly, so batch scoring matches how the predictions are used.
- **Kubernetes, feature store, workflow orchestrator.** The data is small and static, and none of these would solve a problem the scenario has.
- **Other C-MAPSS subsets as separate benchmarks.** FD003 plays the role of the drift incident. FD002 and FD004 (six operating regimes) are a possible extension, not a goal.

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

## Project structure

```
├── data/               # raw + processed data (gitignored, see data/README.md)
├── docs/learning/      # study notes, one level per step of the plan
├── notebooks/          # exploration and decisions, numbered in order
├── src/turbofan_rul/   # importable package: loading, features, evaluation
└── tests/
```

Planned as the production part is built: a pipeline module in `src/turbofan_rul/`, entry-point scripts for training and the weekly run, and a single command that runs the whole pipeline end to end.

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Download the C-MAPSS data (see [data/README.md](data/README.md)), then start with the notebooks:

```bash
uv run jupyter lab
```

Browse the tracked runs (MLflow, stored locally and not committed):

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Run tests:

```bash
uv run pytest
```

## References

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation.* PHM08.
- Jardine, A. K. S., & Tsang, A. H. C. (2013). *Maintenance, Replacement, and Reliability: Theory and Applications* (2nd ed.). CRC Press.
- Huyen, C. (2022). *Designing Machine Learning Systems.* O'Reilly.
- Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS.

## License

MIT
