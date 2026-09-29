# Level 3: Time series and window features

**Reading:** Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed., free at otexts.com/fpp3), ch. 2 (time series graphics, autocorrelation) and ch. 3 (decomposition, moving averages). Section 9.1 for stationarity and differencing.

**Topics:** autocorrelation, stationarity and differencing, moving averages and lag, window slopes, dispersion, per-engine baselines and per-regime normalization, feature redundancy, leakage in feature construction.

## Notes

Study sheet in DIMAP form, same format as levels 1 and 2. This level and level 4 share one notebook, `notebooks/04_features_and_boosting.ipynb`, and this sheet covers its first half: turning each engine's sensor history into features that describe where the engine is and how fast it is moving. The numbers quoted come from a quick check under the notebook 03 protocol (seed-0 cuts, linear model unless stated). The notebook has to reproduce them; if it does not, the notebook is right and this sheet gets corrected.

One result frames the whole level. On the raw sensor readings at the cut, LightGBM scores RMSE 23.8, the same as the linear model's 24.2. On window features, it scores 11.8. The model matters, but the features are what unlock it.

---

### 3.1 Autocorrelation and dependent rows

**Definition.** The autocorrelation at lag $k$ of a series is the correlation between the value at time $t$ and the value at time $t - k$. A series with high autocorrelation changes slowly; consecutive observations carry almost the same information.

**Intuition.** Yesterday's temperature is a good guess for today's. Each new day adds a little information, not a whole new observation.

**Mechanics.** With one series per engine, the autocorrelation is computed within each engine and averaged, never across the concatenated fleet (the last cycle of engine 1 has nothing to do with the first cycle of engine 2). `s.autocorr(k)` in pandas does it for one series; `groupby("unit")` applies it per engine.

**Assumptions.** Rows are in time order within each engine, and the series is long enough for the estimate to be stable (all FD001 engines have at least 128 cycles).

**Practical application.** Notebook 04, section 1. The lag-1 autocorrelation of sensor 4 within an engine is 0.74 raw and 1.00 after a 30-cycle mean. This is the number behind two facts from level 2: rows of an engine are near copies of each other, which is what the row-wise split exploited, and 20,631 rows are worth far fewer independent observations than their count suggests. It also says smoothing does not create information: it removes noise and makes neighbouring rows even more alike.

*Self-check: the 30-cycle mean has autocorrelation 1.00. Why is that expected, and what does it imply for a model that sees consecutive rows of the same engine in training?*

### 3.2 Stationarity, trend and differencing

**Definition.** A series is stationary when its statistical properties (mean, variance, autocorrelation) do not change over time. A trend makes a series non-stationary. Differencing, replacing each value by its change from the previous one, removes a trend.

**Intuition.** Forecasting textbooks want stationarity because their models assume the future looks like the past. Here the opposite is true: the trend *is* the signal. An engine's sensors drift because it is degrading, and the drift is what predicts remaining life. The tools are the same, the goal is inverted: the level does not remove the trend, it measures it.

**Mechanics.** First difference: $\Delta x_t = x_t - x_{t-1}$. On a noisy series the difference is dominated by noise. Its lag-1 autocorrelation goes negative because a positive noise bump is followed by a negative correction. Trend has to be measured over a window, not between two consecutive cycles.

**Assumptions.** The trend of interest is slow compared with the noise, which holds here: degradation moves a sensor by a few standard deviations over 100 cycles, and noise moves it by one standard deviation every cycle.

**Practical application.** Notebook 04, section 1. The first difference of sensor 4 has lag-1 autocorrelation of −0.50: pure noise with a correction pattern, and no use as a feature. The trend is measured with slopes over windows (3.4). The forecasting reflex "difference until stationary" would destroy the target.

*Self-check: why is the first difference of a noisy sensor negatively autocorrelated, and why does that make it a poor feature here?*

### 3.3 Moving averages and their lag

**Definition.** A trailing moving average of length $w$ at cycle $t$ is the mean of cycles $t - w + 1$ to $t$. It estimates the level of the series, with less noise than a single reading.

**Intuition.** Averaging 30 readings cuts the noise by about $\sqrt{30}$, but the average describes the engine as it was 15 cycles ago on average. Level 2, question 2 of section 4, found the cost: near failure, where sensors move fast, a long mean makes the engine look healthier than it is.

**Mechanics.** `groupby("unit")[cols].rolling(w, min_periods=1).mean()`, per engine, trailing only. `min_periods=1` returns a partial mean for the first $w - 1$ cycles; the cut rule of at least 30 cycles observed keeps those out of the evaluation. Within an engine, the 30-cycle mean of sensor 4 has a standard deviation of 5.4 against 8.1 raw: noise removed, but the remaining variation is mostly the degradation trend itself.

**Assumptions.** The choice of $w$ trades noise against lag. Sensors that are noisy relative to their signal (8 and 13, notebook 01) tolerate longer windows; the fastest-moving sensors near failure suffer most from lag.

**Practical application.** Notebook 04, section 2. For the linear model, raw readings plus a 5-cycle mean give RMSE 23.5, raw plus a 30-cycle mean 22.5, raw alone 24.2. The mean helps only in combination with the raw value: the model can then compare the current reading with the recent level, which is a crude trend. Raw plus (raw minus 30-cycle mean) gives exactly the same 22.5, because for a linear model the difference is a linear combination of features it already has. For a tree model the difference is a new, directly splittable quantity.

*Self-check: why do "raw + mean30" and "raw + (raw − mean30)" give identical scores for a linear model, and why might they differ for a tree?*

### 3.4 Slopes over windows

**Definition.** The slope over a window is the coefficient of a straight line fitted to the last $w$ readings against cycle number: the rate of change of the sensor, in units per cycle.

**Intuition.** Two engines with the same sensor 4 reading are not equally close to failure if one is climbing fast and the other is flat. The reading says where the engine is, the slope says where it is going.

**Mechanics.** Least squares on the window: with $x = 0, 1, \dots, w - 1$ and $\bar x$, $\bar y$ their means,

$$
\hat\beta = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^2}.
$$

Computed per engine over a trailing window. Short windows give noisy slopes: the noise of the readings divided by a small spread of $x$. Long windows give stable slopes that lag, like the mean. A rolling `apply` with `np.polyfit` is slow on 20,000 rows; a closed-form rolling covariance is faster and belongs in `features.py`.

**Assumptions.** Degradation is roughly linear inside the window. Near failure it accelerates, so a 30-cycle slope underestimates the current rate, but it still ranks engines correctly.

**Practical application.** Notebook 04, section 2. The 30-cycle slope is the strongest single window feature for the linear model: raw plus slope30 gives RMSE 18.9, against 23.5 for slope10 (too noisy to help). It halves the PHM08 score, from 1,536 to 644. Slopes give the linear model what it lacked in level 2: a way to see acceleration without bending.

*Self-check: why does the 10-cycle slope help so little when the 30-cycle slope helps so much, and what would you expect from a 60-cycle slope?*

### 3.5 Dispersion within a window

**Definition.** The rolling standard deviation over a window measures how much the readings scatter around their local level.

**Intuition.** Some failure modes show up as instability before they show up as drift: a bearing that starts to vibrate. In C-MAPSS the degradation is a smooth drift with constant sensor noise, so dispersion carries little extra.

**Mechanics.** `rolling(w, min_periods=2).std()` per engine. During a fast drift the window std rises, because the window spans a range of levels, so the feature is partly a slope in disguise.

**Assumptions.** Useful when the noise level itself changes with health. That is not how the simulation was built.

**Practical application.** Notebook 04, section 2. Raw plus a 30-cycle std gives RMSE 22.9: a small gain, less than the mean and far less than the slope. It stays in the feature set because the tree model can use it cheaply, but it is not where the signal is.

*Self-check: why does the rolling std rise during a drift even when the sensor noise is constant?*

### 3.6 Deviation from the engine's own baseline

**Definition.** For each engine, the baseline of a sensor is its mean over the first cycles of life, when the engine is healthy. The deviation feature is the current smoothed reading minus that baseline.

**Intuition.** Every engine starts with its own initial wear (readme), so absolute readings carry an engine-specific offset. A sensor 7 reading of 553 means different things for an engine that started at 554 and one that started at 551. Measuring each engine against its own healthy self removes the offset. Notebook 01 found this for sensors 9 and 14, whose trend sign differs by engine; here it is applied to all sensors.

**Mechanics.** Baseline: mean of cycles 1 to 20 per engine, `groupby("unit").transform(lambda s: s.iloc[:20].mean())`. Deviation: 10-cycle mean minus baseline. For sensors 9 and 14 the absolute value is used, because the direction is engine-specific and only the distance from the baseline is informative.

**Assumptions.** Every engine is observed from cycle 1, and its first 20 cycles are healthy. Both hold in C-MAPSS train and test (the shortest test history is 31 cycles). In a real fleet an engine may enter monitoring mid-life, and the baseline would have to come from the fleet or from a maintenance record. This is the same idea as per-regime normalization for FD002 and FD004: there, the offset comes from the operating condition rather than the engine, and the baseline is the mean of the regime.

**Practical application.** Notebook 04, section 2. The deviation features alone give RMSE 20.9, better than any raw-plus-window set except slope30; raw plus deviation gives 19.4. For LightGBM they dominate the gain importance (sensor 15 deviation alone carries 36% of it, level 4). The absolute deviations of sensors 9 and 14 add little to the linear model (23.4) but rank second and fifth in permutation importance for LightGBM: their information is non-linear.

*Self-check: what breaks if an engine's first 20 cycles are not healthy, and how would you detect it?*

### 3.7 Feature sets, redundancy and regularization

**Definition.** Combining 12 sensors, three window lengths and several statistics produces about a hundred features, many of them near-duplicates: a sensor's 5-cycle mean, 10-cycle mean and raw reading are strongly correlated.

**Intuition.** Redundant features do not hurt a tree model much (it picks one and ignores the rest), but they make importance rankings unreliable (credit is split arbitrarily, level 4) and slow everything down. For a linear model they raise variance, which is when regularization earns its place.

**Mechanics.** Level 2 predicted that ridge and lasso would start to matter with around a hundred correlated features. Check it under the protocol: OLS against `RidgeCV` with the penalty chosen inside the training folds.

**Assumptions.** Regularization helps when the number of features is large relative to the number of independent units. Here the ratio is about 100 features to 80 training engines, but each engine contributes 200 rows, and the effective sample is somewhere in between.

**Practical application.** Notebook 04, section 2. With all 99 features, OLS gives RMSE 18.8 and ridge 18.8: still no variance to remove, because 16,000 rows, even if correlated, are enough for a hundred coefficients. The linear model's remaining problem is still bias: it is late by 16 to 17 cycles between 25 and 75 cycles from failure, against 3 to 4 for LightGBM on the same features. On the tree side, 62 of the 99 features have a permutation importance near zero: the set can be pruned to a third without losing anything, which matters for the pipeline in level 8.

*Self-check: the linear model has 99 features and 80 training engines, yet ridge does not help. Which assumption of "regularize when p is large relative to n" is being violated?*

### 3.8 Leakage in feature construction

**Definition.** A feature leaks when its value at cycle $t$ uses information that would not exist at cycle $t$ in operation: later cycles of the same engine, or statistics computed over engines the model is evaluated on.

**Intuition.** Level 2 question 1 of section 4: a centred window leaks the future. The other leak is subtler: a fleet-wide statistic (the mean of a sensor over all engines, a scaler fitted on all rows) computed before the split includes the validation engines.

**Mechanics.** Three rules. Windows are trailing (`rolling` in pandas is trailing by default). Everything is computed per engine with `groupby("unit")`, so no window crosses from one engine into the next. Per-engine baselines use only that engine's own first cycles, which exist at any cut after cycle 20. Fleet-level statistics (scalers, regime means for FD002) are fitted inside the training folds; a `Pipeline` with `StandardScaler` does this automatically.

**Assumptions.** The cut rule guarantees at least 30 observed cycles, so every trailing window and baseline is complete at evaluation time.

**Practical application.** Notebook 04, section 1 builds the features once for the whole training set, which is safe only because every feature is trailing and per-engine. Section 3 repeats the level 2 leakage check with LightGBM: a row-wise split moves it from 11.8 to 8.5. Trees memorize less than nearest neighbours (level 2, question 3 of section 4) but the leak is there, and it would be larger with deeper trees.

*Self-check: the per-engine baseline uses cycles 1 to 20. A cut at cycle 15 would use a baseline that includes cycles 16 to 20. Which protocol rule prevents this, and what would you do for an engine that enters the fleet mid-life?*

---

## Exercise

First half of `notebooks/04_features_and_boosting.ipynb`. Code goes in `src/turbofan_rul/features.py`, with tests.

1. **Autocorrelation and differencing** (3.1, 3.2): lag-1 autocorrelation of two sensors within engines, raw, smoothed and differenced. One cell.
2. **Feature families** (3.3 to 3.6): trailing means (5, 10, 30), differences from the 30-cycle mean, slopes (10, 30), 30-cycle std, deviation from the per-engine baseline, absolute deviations for sensors 9 and 14, age. Implemented in `features.py` with a closed-form rolling slope.
3. **Which features carry signal**: the linear model on raw plus each family under the protocol, reported with the 30-draw mean and per-fold scores. Expected order: slope30 and deviation first, means and std last.
4. **All features**: OLS against ridge, to close the regularization question from level 2.
5. **Leakage rules** written into `features.py` docstrings and checked by a test: a feature at cycle $t$ must not change when later cycles are removed.

## Open questions

- The baseline uses the first 20 cycles. Is 20 the right length, and should it be a fraction of the minimum history instead of a constant?
- Slopes are computed on raw readings. Would slopes of the smoothed series be less noisy at short windows, or just lag more?
- For FD002 and FD004, per-regime normalization replaces the per-engine baseline. Can the two be combined (engine baseline within regime), and is there enough data per engine and regime to estimate it?
- 62 of 99 features have near-zero permutation importance. Should the pipeline in level 8 carry the full set for robustness, or the pruned set for simplicity?
