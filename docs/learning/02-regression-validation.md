# Level 2: Regression and validation

**Reading:** James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, ch. 3, 5, 6. Bergmeir & Benítez (2012), *On the use of cross-validation for time series predictor evaluation*.

**Topics:** linear regression and its assumptions, heteroscedasticity, bias-variance, L1/L2 regularization, group-wise and temporal cross-validation.

## Notes

<!-- What I learned, in my own words -->

## Exercise

Linear baseline on each sensor's last value, split by engine; measure how the error changes with a row-wise split (leakage demo).

Notebook: `notebooks/03_linear_baseline.ipynb`

Build the development protocol that every later model uses: engine-wise folds on the training engines, out-of-fold RMSE and PHM08, and the clock from notebook 02 recomputed out-of-fold as the baseline. Decide how cut points are sampled on training engines so that the evaluation mimics the test set (one prediction per engine at a truncated history). The FD001 test set is not touched here: it is used once, at the end of the project. The out-of-fold predictions are also what the level 8 simulation replays. This is also where the production groundwork starts: log the clock and the linear baseline as the first MLflow runs, and set up CI with the existing tests.

## Open questions

-
