"""Reference models that use no sensor data."""

import numpy as np
from scipy import stats


class WeibullClock:
    """Age-only RUL model: the median remaining life of an engine that has survived to its age.

    Fits a three-parameter Weibull to complete lifetimes (notebook 02). For an engine of age t,
    the prediction is the s that solves S(t + s) = 0.5 * S(t): half of the engines that reach
    age t fail within s more cycles.
    """

    def fit(self, lifetimes) -> "WeibullClock":
        self.dist_ = stats.weibull_min(*stats.weibull_min.fit(np.asarray(lifetimes, dtype=float)))
        return self

    def predict(self, ages) -> np.ndarray:
        ages = np.asarray(ages, dtype=float)
        return self.dist_.isf(0.5 * self.dist_.sf(ages)) - ages
