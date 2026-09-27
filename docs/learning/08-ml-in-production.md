# Level 8: Running the model in production

**Reading:** Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022), chapters 7 to 9 (deployment and batch prediction, data distribution shifts and monitoring, continual learning). Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems* (NeurIPS). Rabanser, Günnemann & Lipton (2019), *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift* (NeurIPS). MLflow documentation, tracking and model registry.

**Topics:** batch versus online prediction, reproducible pipelines, experiment tracking and model registry, CI for ML code, covariate shift versus concept drift, drift detection on inputs and predictions, delayed labels, retraining triggers, backtesting a decision system by replay.

## Notes

<!-- What I learned, in my own words -->

## Exercise

Run the whole system as it would run in operation, and measure it.

1. A weekly batch job: ingest the new cycles, compute features, score with the registered model, produce the shop plan. One command runs it end to end.
2. A replay of a fleet the model has never seen, week by week, applying each plan and recording what actually happens: planned removals, unplanned failures, wasted cycles, total cost. Compare against the clock policy and the MSE model.
3. A drift incident: FD003 engines enter the replayed fleet partway through. Detect the shift, retrain, and show the cost before and after.

Code: `src/turbofan_rul/` (pipeline module) and entry-point scripts.

## Open questions

- Which engines form the replayed fleet? Options: a set of training engines held out from the start, or out-of-fold predictions so every engine is replayed by a model that never saw it. The choice affects how the level 3 validation split is built.
- Labels arrive late and are censored. The true RUL of an engine is only known when it fails, and an engine removed for maintenance never reaches failure: it is a right-censored observation, the case from level 1. How should retraining use those engines?
- Which drift signal fires first on FD003: input distributions, prediction distributions, or realized errors (which only arrive once engines fail)?
