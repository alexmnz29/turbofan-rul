# Fundamentals behind notebook 01, in DIMAP form

Each entry follows DIMAP: Definition, Intuition, Mechanics, Assumptions, Practical application. The "practical" part always points to where the element appears in `notebooks/01_eda_fd001.ipynb`, so the notebook doubles as the exercise sheet. The closing question in each entry is the self-check: if you cannot answer it without looking, reread the entry.

---

## Block 1. Pandas: grouped data

### 1.1 `groupby` and its three outputs: `agg`, `transform`, `apply`

**Definition.** `groupby` splits a DataFrame into groups by the values of one or more columns and lets you compute something per group. `agg` returns one row per group, `transform` returns a result the same length as the input, and `apply` returns whatever the function returns.

**Intuition.** Think of a class register. `agg` is the teacher writing one line per class ("class 3B: average 7.2"). `transform` is stamping each student's sheet with their class average, so every student still has their own row but now carries a group-level number. `apply` is handing each class to an assistant and taking back whatever they produce, which may be a number, a table, or a plot.

**Mechanics.** Split, apply, combine. `agg(f)` calls `f` on each group's column and stacks the scalars, indexed by group key. `transform(f)` calls `f` per group, expects a same-length output, and puts the values back at their original row positions (index-aligned). `apply(f)` calls `f` on each group as a sub-DataFrame; if all results are Series with the same index, pandas builds a DataFrame with groups as rows. Group keys become the index of the result unless you use `as_index=False`.

**Assumptions.** The grouping column must identify the unit you care about, and the per-group function must be valid on every group (an empty or one-row group can break correlations or stds). `transform` requires the function to return one value per input row or a scalar to broadcast.

**Practical application.** Section 1: `df.groupby("unit")["cycle"].agg(["min", "max", "count"])` gives one row per engine. Section 3a: `df.groupby("unit")["cycle"].transform("max") - df["cycle"]` builds `cycles_to_failure` on every row. Section 3b: `groupby("unit").apply(lambda g: g[sensors].corrwith(g["cycle"]))` yields a 100 x 15 table of correlations. In feature engineering, almost every rolling statistic is a `groupby(...).transform(...)`.

*Self-check: before running a groupby line, can you say the shape of its output?*

### 1.2 Index alignment

**Definition.** Arithmetic and concatenation between pandas objects match rows by index label, not by position.

**Intuition.** Two spreadsheets with employee IDs: subtracting "salary this year" from "salary last year" should match by ID, not by row number. Pandas does that by default, which is safe when the IDs agree and silently wrong when they do not.

**Mechanics.** For `a - b`, pandas takes the union of the two indexes, aligns both to it, and computes elementwise; labels present in only one side produce NaN. `pd.concat([...], axis=1)` does the same union on the row index and places the columns side by side. `reset_index(drop=True)` replaces labels with positions, which is how you deliberately force positional matching.

**Assumptions.** The indexes mean the same thing on both sides. After sorting, filtering, or grouping, the index may no longer be the default range, and that changes what "aligned" means.

**Practical application.** Section 3b: `smoothed - baseline` subtracts per-row because both were produced by `transform` on `df` and share its index. `pd.concat([df[["unit", "cycle"]], smoothed], axis=1)` glues columns without any explicit key. In Section 3a, `rolled.mean().reset_index(level=0, drop=True)` is needed inside the `features.py` stub because the rolling result carries a (unit, row) MultiIndex that must be flattened before assignment.

*Self-check: what happens to `a - b` if `b` has been sorted by a different column first?*

### 1.3 Vectorized operations and broadcasting

**Definition.** Applying an operation to a whole column or table at once, in compiled code, instead of looping in Python. Broadcasting is the rule that lets a smaller object (scalar or Series) combine with a larger one.

**Intuition.** Instead of telling each cashier separately to apply a 10% discount, you announce it once and every till does it. Broadcasting is the announcement reaching every till.

**Mechanics.** `df[cols].std()` computes one std per column in C. A DataFrame combined with a Series aligns the Series index to the DataFrame's *columns* and repeats it down every row. `np.sign(corr_by_unit) == np.sign(corr_by_unit.median())` compares a 100 x 15 table with a length-15 Series: the Series is matched to the 15 columns and broadcast across the 100 rows. `.mean()` on the boolean result gives the fraction of True per column.

**Assumptions.** Column names (or index labels) must match for the alignment to be meaningful. A Series whose index does not match the columns produces a table full of NaN without an error.

**Practical application.** Section 3a: `sensor_summary` is built from three vectorized reductions. Section 3b: `sign_consistency` is one broadcast comparison plus a mean. Section 4: `(z * direction).mean(axis=1)` multiplies each column by its own sign (Series aligned to columns), then averages across columns for each row. If you find yourself writing `for col in df.columns:`, ask whether a single vectorized line does it.

*Self-check: in `z * direction`, which axis does `direction` align to, and why is that the one you want?*

### 1.4 Rolling windows and `min_periods`

**Definition.** A rolling window computes a statistic over the last `n` observations at every position. `min_periods` is the minimum number of observations required before a value is returned.

**Intuition.** A 20-day moving average of a stock price. On day 3 you only have 3 prices: with `min_periods=1` you average what you have; with `min_periods=20` you report nothing until day 20.

**Mechanics.** `s.rolling(n, min_periods=k).mean()` returns a Series aligned with `s`; positions with fewer than `k` prior values are NaN (or, with `k=1`, averages of however many exist). Windows are trailing by default, so the value at position `t` uses `t-n+1 ... t`. Inside `groupby(...).transform(...)`, the window restarts at every group boundary.

**Assumptions.** Rows are in time order within each group. The window is applied per group, never across groups (a rolling mean over the whole `df` would blend the last cycles of engine 1 with the first cycles of engine 2). The first `n-1` values are less smoothed than the rest, which matters for any downstream correlation or threshold.

**Practical application.** Section 3a: `sub[col].rolling(20, min_periods=1).mean()` for the thick plot lines; the "hook" at the start of each curve is the `min_periods=1` effect. Section 3b: the same smoothing inside `transform` before correlating. The level 3 exercise multiplies this by three window sizes and three statistics.

*Self-check: why must the rolling call sit inside `groupby("unit")` rather than on `df[col]` directly?*

### 1.5 MultiIndex, `quantile`, and `unstack`

**Definition.** A MultiIndex is an index with more than one level. `unstack` pivots the innermost level from rows into columns.

**Intuition.** Sales by (region, month) stored as a long list versus as a grid with regions down the side and months across the top. Same data, two shapes; `unstack` moves from the list to the grid.

**Mechanics.** `df.groupby("ctf")["health_index"].quantile([0.1, 0.5, 0.9])` returns a Series indexed by (ctf, quantile). `.unstack()` turns the quantile level into three columns named 0.1, 0.5, 0.9, one row per ctf. `.loc[:250]` then slices the row index. The inverse operation is `stack`.

**Assumptions.** The (outer, inner) pairs must be unique for `unstack` to pivot without ambiguity. Column names coming out of `unstack` are the level values themselves (floats here), so you index them as `fleet[0.5]`, not `fleet["0.5"]`.

**Practical application.** Section 4: `fleet` is exactly this pivot, and `fleet[0.5]` is the fleet median at each cycles-to-failure. The Section 1 `agg(["min", "max", "count"])` produces plain columns, but `agg` over several columns produces a MultiIndex on the columns that you would flatten the same way.

*Self-check: what would `fleet` look like if you forgot `.unstack()`, and why would `fleet[0.5]` then fail?*

---

## Block 2. Descriptive statistics

### 2.1 Mean, median, and skewness

**Definition.** The mean is the arithmetic average; the median is the middle value; skewness measures the asymmetry of a distribution around its center.

**Intuition.** Household income: a few very rich households pull the mean up, but the median household does not feel richer. When mean > median, there is a tail on the right.

**Mechanics.** Mean = sum / n. Median = 50th percentile. Sample skewness is the standardized third central moment, roughly mean of ((x - mean) / std)^3; positive means a right tail, negative a left tail, zero is symmetric. The mean is sensitive to extreme values; the median is not.

**Assumptions.** Skewness as a single number describes unimodal distributions well. For bimodal ones it can be near zero while the shape is far from symmetric (see 2.4).

**Practical application.** Section 2: lifetimes have mean 206, median 199, skewness 1.03, so a right tail of long-lived engines. This tells you a clock-based predictor is biased and that a validation split should check the tail is represented. The Weibull fit in level 1 is the parametric version of this description.

*Self-check: if mean < median, which way is the tail?*

### 2.2 Standard deviation and coefficient of variation

**Definition.** Standard deviation is the typical distance of values from their mean, in the variable's own units. The coefficient of variation (CV) is std divided by the absolute mean: dispersion relative to level, unitless.

**Intuition.** A 1 kg fluctuation is huge for a cat and nothing for an elephant. Std says "1 kg"; CV says "5% of body weight".

**Mechanics.** std = sqrt(mean((x - mean)^2)) with an n-1 denominator in pandas. CV = std / |mean|. CV is undefined or unstable when the mean is near zero, which is why it is not used for `setting_1` and `setting_2` (means around 1e-5).

**Assumptions.** CV only makes sense for ratio-scale variables with a meaningful zero and a mean well away from it. Comparing stds across variables assumes they share units and scale, which sensors do not.

**Practical application.** Section 3a: `sensor_summary["cv"]` ranks the 21 sensors on equal footing; sensor_9 (std 22, CV 0.002) is far less variable relative to its level than sensor_4 (std 9, CV 0.006). Section 4 uses std as the scaling factor of the z-score for the same reason.

*Self-check: why is CV a bad metric for `setting_1`?*

### 2.3 Pearson versus Spearman correlation

**Definition.** Pearson measures the strength of a linear relationship between two variables. Spearman is Pearson computed on the ranks of the values, so it measures monotonic association.

**Intuition.** Pearson asks "do the points sit near a straight line?". Spearman asks "when one goes up, does the other go up, whatever the shape?". An exponential curve is perfectly monotonic (Spearman 1) but not a straight line (Pearson < 1).

**Mechanics.** Pearson r = cov(x, y) / (std_x std_y), in [-1, 1]. Spearman rho: replace each x and y by its rank (1 to n), then compute Pearson on the ranks. Ranks make it robust to outliers and to any monotonic transform of the data. If one variable is constant, its std is zero and the correlation is undefined (NaN, with a warning).

**Assumptions.** Both require paired observations. Pearson assumes the relationship is approximately linear and is sensitive to outliers. Spearman assumes monotonicity is what you care about; it is blind to the shape of the curve.

**Practical application.** Section 3b: `corrwith(g["cycle"], method="spearman")` because the degradation curves are exponential near failure. The `ConstantInputWarning` on sensor_6 is the undefined case. A Pearson version would understate the trend for the sensors with the sharpest final rise.

*Self-check: two sensors both have Spearman 0.95 with age; one is linear and the other exponential. What would Pearson say about each?*

### 2.4 Distributions versus summary statistics

**Definition.** A summary statistic compresses a distribution into one number. A distribution is the full pattern of values. Bimodality is a distribution with two separate peaks.

**Intuition.** Average shoe size in a mixed-gender group tells you the size that fits almost nobody. The mean is fine; it just does not describe anyone.

**Mechanics.** Mean and median of a bimodal distribution land between the peaks or on the larger one. Detecting bimodality needs the shape: a histogram, a boxplot with an unusually wide box, a kernel density plot, or a mixture model. A quick numeric hint is a large gap between the median and the 25th or 75th percentile.

**Assumptions.** Any one-number summary assumes the distribution is unimodal and roughly symmetric enough for that number to be representative. Check that before trusting it.

**Practical application.** Section 3b: sensors 9 and 14 have median correlation 0.93 (looks excellent) but boxes spanning -0.75 to 0.97 and sign consistency around 0.65. The boxplot is what revealed two populations of engines. The rule to keep: when a summary drives a decision, look at the distribution it summarizes at least once.

*Self-check: name two plots and one statistic that would have exposed the bimodality of sensor_9.*

### 2.5 Standardization and k-sigma thresholds

**Definition.** Standardization (z-score) rescales a variable to (x - reference) / std, so values are in units of standard deviations. A k-sigma threshold flags values more than k stds from a reference as unusual.

**Intuition.** A thermometer reading of 38.5 means nothing until you know normal is 36.5 with a spread of 0.3; then it is 6.7 spreads above normal, which is clearly not noise. Standardizing is expressing everything as "how many spreads from normal".

**Mechanics.** z = (x - mu) / sigma. With mu and sigma taken from a healthy reference, z near 0 is normal and |z| > 3 happens about 0.3% of the time under a normal distribution, hence the 3-sigma convention. Averaging z-scores of several variables (after flipping signs so "worse" is positive) produces a composite index on a common scale.

**Assumptions.** The reference period is truly healthy and long enough to estimate mu and sigma. The 0.3% interpretation assumes normality; with heavy tails or autocorrelation, 3-sigma fires more often than expected. Scaling by a fleet-wide std (as in the notebook) rather than a per-engine std mixes within- and between-engine variability, which is a simplification.

**Practical application.** Section 4: `z = (smoothed - baseline) / df[raw_keep].std()`, then `health_index = (z * direction).mean(axis=1)`, then `threshold = healthy.mean() + 3 * healthy.std()`. The sensitivity table shows the onset moving from 153 to 137 cycles as k goes from 2 to 5. This is the core of statistical process control and of most simple anomaly detectors.

*Self-check: what changes in the onset estimate if the "healthy" reference window includes the early part of the degradation phase?*

### 2.6 Percentiles and the boxplot

**Definition.** The p-th percentile is the value below which p% of the data fall. A boxplot draws the 25th, 50th, and 75th percentiles as a box, whiskers out to the most extreme points within 1.5 times the interquartile range, and individual points beyond that.

**Intuition.** The box is where the middle half of the data lives; the whiskers are the reasonable range; the dots are the surprises. A wide box means the data disagree with each other.

**Mechanics.** IQR = p75 - p25. Whisker limits are p25 - 1.5 IQR and p75 + 1.5 IQR, clipped to actual data points. Anything outside is drawn as an outlier. A band between the 10th and 90th percentiles, as in Section 4, is an alternative that shows where 80% of engines sit at each time step.

**Assumptions.** The 1.5 IQR outlier rule is a convention, not a test. Percentiles need enough points to be stable; with 100 engines the 10th percentile is the 10th-smallest value, which is fine, but with 10 engines it would be one point.

**Practical application.** Section 3b: one boxplot per sensor, 100 points each; the width of the box is the visual encoding of `sign_consistency`. Section 4: `fill_between(fleet[0.1], fleet[0.9])` is a percentile band over time. Section 2: `describe()` reports the 25/50/75 percentiles of lifetimes.

*Self-check: sensor_13 shows several outliers near zero but a tight box. What does that say about the fraction of engines where the trend is weak?*

---

## Block 3. Matplotlib

### 3.1 The object-oriented interface

**Definition.** Matplotlib has two APIs: the stateful `plt.plot(...)` style and the object-oriented style where you hold a `Figure` and one or more `Axes` objects and call methods on them.

**Intuition.** `plt.*` is like shouting instructions into a room and hoping the right canvas hears. `ax.plot(...)` is handing the brush to a specific canvas. With one plot it does not matter; with a 5 x 3 grid it is the only way to stay sane.

**Mechanics.** `fig, axes = plt.subplots(nrows, ncols, figsize=..., sharex=...)` creates the grid; `axes.flat` iterates it in reading order. Each `ax` has `plot`, `hist`, `boxplot`, `fill_between`, `axvline`, `set(...)` for labels and title, `legend`, `invert_xaxis`. `ax.twinx()` creates a second Axes sharing the x-axis with its own y-axis; legends from both must be merged by hand (`get_legend_handles_labels`). `plt.tight_layout()` fixes overlaps; `plt.show()` renders.

**Assumptions.** One `ax` per panel. Anything you want to control per panel (title, limits, legend) goes through that `ax`, not through `plt`.

**Practical application.** Section 3a: `zip(axes.flat, candidate_sensors)` pairs 15 panels with 15 sensors. Section 4: `twinx` puts RUL labels and the health index on one chart with two scales; `fill_between` draws the percentile band; `axvline` marks onset and the literature cap.

*Self-check: in the Section 4 label plot, why does `ax1.legend()` alone not show the health index line?*

### 3.2 Choosing the x-axis that aligns what you compare

**Definition.** The x-axis defines what "same position" means across series. Choosing it is an analytic decision, not a cosmetic one.

**Intuition.** Comparing how runners finish a race, you align on the finish line, not on the starting gun, if the races had different lengths. Engines with lifetimes of 128 and 362 cycles are races of different length.

**Mechanics.** Plotting against `cycle` aligns engines at birth; degradation then happens at different x positions and the curves do not overlap. Plotting against `cycles_to_failure` (with `invert_xaxis()` so time still flows left to right) aligns them at death, so any pattern tied to the end of life lines up. The same choice drives the fleet aggregation in Section 4: grouping by cycles-to-failure is what makes a "fleet median at 100 cycles before failure" meaningful.

**Assumptions.** You know the alignment event (failure) for every series. In the test set you do not, which is precisely why the model is needed there; this alignment is an EDA tool, not a feature you can compute at prediction time.

**Practical application.** Sections 3a and 4. The bend appearing at the same x for all engines is the single most important visual result in the notebook and exists only because of this axis choice.

*Self-check: why can you not use `cycles_to_failure` as a model input?*

---

## Block 4. Python and notebook engineering

### 4.1 Floating-point arithmetic

**Definition.** Computers store real numbers in binary floating point with finite precision, so most decimals are approximations and arithmetic accumulates tiny errors.

**Intuition.** 1/3 in decimal is 0.3333... and you have to cut it somewhere. Binary has the same problem with 0.1. Add enough rounded numbers and the error becomes visible.

**Mechanics.** `0.1 + 0.2 == 0.3` is False in Python (the sum is 0.30000000000000004). The std of a constant column involves subtracting a mean that is itself rounded, so the result can be ~1e-15 instead of 0. Safe comparisons use a tolerance (`abs(a - b) < 1e-9`, `np.isclose`) or an exact integer criterion where one exists.

**Assumptions.** Any test of the form `float == value` is suspect unless the values came from integer arithmetic. Sorting, ranking, and counting distinct values are exact operations and can replace approximate ones.

**Practical application.** Section 3a: `std == 0` caught 4 constant sensors instead of 6; `nunique() == 1` is the exact definition and caught all 6. The bug would have silently dropped two sensors from the plot grid because of how `zip` truncates (see 4.3).

*Self-check: give two ways to test "this column is constant" that do not depend on floating-point equality.*

### 4.2 `assert` as a contract

**Definition.** `assert condition, message` raises `AssertionError` with the message if the condition is False. It states an assumption the rest of the code depends on.

**Intuition.** A sign at the entrance of a bridge: "max 10 tonnes". The bridge does not check; the sign makes the assumption visible and gives you something to blame when it breaks. `assert` is the sign that also checks.

**Mechanics.** Runs at the point where it is written; on failure it stops execution and prints the message. Cheap conditions (counts, uniqueness, monotonicity) are fine in notebooks and scripts. Note that `python -O` strips asserts, so production code uses explicit `if ...: raise ValueError(...)` instead; the loader in `data_loading.py` already does this.

**Assumptions.** The condition is genuinely required downstream, not merely expected. Asserting things that may legitimately vary makes the notebook brittle.

**Practical application.** Section 1: four asserts encode "100 engines, no NaNs, cycles are 1..n contiguous and increasing". `add_rul_labels` and every rolling feature silently depend on the last three. If FD002 data had a gap in one engine, these lines are where you would find out.

*Self-check: which downstream computation would give wrong answers without an error if an engine had cycles 1..50 and 52..200?*

### 4.3 List comprehensions and `zip`

**Definition.** A list comprehension builds a list from an iterable in one expression, optionally with a filter. `zip` pairs elements from several iterables position by position, stopping at the shortest.

**Intuition.** A comprehension is "the list of X for each Y where Z" written the way you would say it. `zip` is a zipper: it closes tooth by tooth and stops when one side runs out.

**Mechanics.** `[s for s in SENSOR_COLS if s not in constant_sensors]` filters names. `[f"{s}_mean_{w}" for s in sensors]` builds names. `zip(axes.flat, candidate_sensors)` yields (axis, sensor) pairs; if there are 15 axes and 17 sensors, the last two sensors are never drawn and nothing warns you. Python 3.10+ has `zip(..., strict=True)` which raises on length mismatch.

**Assumptions.** With plain `zip`, you are asserting the lengths match. If you have not checked, use `strict=True` or assert the lengths first.

**Practical application.** Section 3a: the 5 x 3 grid assumes exactly 15 candidate sensors. With the floating-point bug there were 17, and `zip` would have hidden it. Comprehensions appear in every feature-name construction in `features.py`.

*Self-check: rewrite the Section 3a `zip` so that a mismatch between panels and sensors fails loudly.*

### 4.4 Importing from your own package (`src` layout)

**Definition.** A `src` layout keeps importable code in `src/<package>/`, installed into the environment so notebooks and tests can `import turbofan_rul` from anywhere.

**Intuition.** The notebook is the lab bench; the package is the toolbox. Tools you reach for repeatedly belong in the toolbox, with a fixed place and a test, not loose on the bench.

**Mechanics.** `pyproject.toml` declares the package and build backend; `uv sync` installs it in editable mode into `.venv`, so edits to `src/` are visible on the next import (restart the kernel or use autoreload). The notebook then does `from turbofan_rul.data_loading import load_subset`. Constants like `SENSOR_COLS` and `RUL_CAP` live in `config.py` so every notebook agrees on them.

**Assumptions.** The kernel runs from the project's `.venv` (Jupyter launched via `uv run`). Code that has stabilized moves into the package; code still being explored stays in the notebook.

**Practical application.** Section 1 imports `load_subset`; Section 4 imports `RUL_CAP` and `add_rul_labels`. The sensor lists decided in Section 3 are candidates to move into `config.py` once you confirm them on the other subsets.

*Self-check: what would break if you opened the notebook with a system Python kernel instead of the project's `.venv`?*

### 4.5 Method chaining

**Definition.** Calling methods one after another on the result of the previous call, in one expression.

**Intuition.** A recipe read as one sentence: "take the flour, sift it, mix with eggs, rest 20 minutes". Each step takes the previous result; you do not need a bowl for every intermediate.

**Mechanics.** Read left to right and track the type and shape after each dot. `df.groupby("cycles_to_failure")["health_index"]` is a grouped Series; `.quantile([...])` is a Series with a 2-level index; `.unstack()` is a DataFrame with 3 columns; `.loc[:250]` is the same DataFrame with fewer rows. Pandas methods return new objects, so chains are side-effect free unless you use `inplace=True` (avoid it).

**Assumptions.** You know the return type of every method in the chain. If you do not, break the chain, assign intermediates, and inspect `.shape` and `.head()`; join it back once it works.

**Practical application.** Sections 2, 3b, and 4 all use chains of two to five steps. The habit that keeps them readable is one chain per idea, with a comment only when a step's output shape is not obvious.

*Self-check: write down the type and shape after each step of the `fleet` chain in Section 4.*

---

## How to use this file

Read one entry, close it, and explain the element aloud in the five DIMAP steps. Then open the notebook, find the line where it appears, and change something (a window size, a percentile, the x-axis) to see the effect. An element counts as known when you can predict the effect before running the cell.
