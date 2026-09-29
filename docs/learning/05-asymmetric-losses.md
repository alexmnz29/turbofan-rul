# Level 5: Asymmetric losses and uncertainty

**Reading:** Koenker & Hallock (2001), *Quantile Regression* (Journal of Economic Perspectives). Saxena et al. (2008), *Damage propagation modeling for aircraft engine run-to-failure simulation* (defines the PHM08 asymmetric score).

**Topics:** the incoherence of training with MSE while evaluating with asymmetric cost, quantile regression, bootstrap intervals, calibration.

## Notes

<!-- What I learned, in my own words -->

## Exercise

Define the maintenance cost function from the README scenario (unplanned failure = 10 shop visits, each wasted cycle = 1/200 shop visit), retrain with it or with a quantile objective, and compare against the MSE model. The quantile outputs feed the planning step in level 7, so calibration matters as much as the score. This is where the project stops being a tutorial.

Notebook: `notebooks/05_asymmetric_cost.ipynb` · Code: `src/turbofan_rul/evaluation.py`

## Open questions

-
