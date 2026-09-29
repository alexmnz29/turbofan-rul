# Level 2: Regression and validation

**Reading:** James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (2nd ed., 2021), ch. 3 (linear regression), ch. 5 (resampling methods) and ch. 6 (linear model selection and regularization); section 2.2 for the bias-variance trade-off. Bergmeir & Benítez (2012), *On the use of cross-validation for time series predictor evaluation*, Information Sciences.

**Topics:** linear regression and its assumptions, residual diagnostics, heteroscedasticity, bias-variance, L1/L2 regularization, cross-validation and out-of-fold predictions, grouped data and leakage, evaluating like the test set.

## Notes

Study sheet in DIMAP form (Definition, Intuition, Mechanics, Assumptions, Practical application), same format as levels 0 and 1. The practical paragraphs point to sections of `notebooks/03_linear_baseline.ipynb`, which does not exist yet. The numbers quoted come from a quick check on the FD001 training engines (12 useful sensors from notebook 01, row values, 5 engine-wise folds). Notebook 03 has to reproduce them, and if it does not, the notebook is right and this sheet gets corrected. Rewrite each entry in your own words once you have run the section.

---

### 2.1 Linear regression and least squares

**Definition.** A linear model predicts the target as a weighted sum of the inputs,

$$
y = \beta_0 + \sum_{j=1}^{p} \beta_j x_j + \varepsilon,
$$

and ordinary least squares (OLS) picks the coefficients that minimise the residual sum of squares $\mathrm{RSS} = \sum_i (y_i - \hat y_i)^2$. In matrix form the solution is $\hat\beta = (X^\top X)^{-1} X^\top y$.

**Intuition.** Fit the flattest possible surface through a cloud of points so that the vertical distances, squared, are as small as possible. Each coefficient says how much the predicted RUL moves when one sensor moves by one unit and the others stay where they are.

**Mechanics.** Standardise the sensors first (subtract the mean, divide by the standard deviation). Predictions do not change, but coefficients become comparable: each one is "cycles of RUL per standard deviation of that sensor". Read "the others stay where they are" with care: when two sensors move together in practice, holding one fixed while the other moves describes a situation that never happens in the data.

**Assumptions.** The relationship is linear in the inputs, and the errors behave as described in 2.2. Nothing here requires normality to fit the model; normality only matters for the textbook confidence intervals.

**Practical application.** Notebook 03, first model: the 12 sensors kept in notebook 01, raw row values, target capped at 125. Out-of-fold RMSE over all training rows is about 22.9. The coefficient signs agree with the physics from notebook 01: sensors that rise with degradation (2, 3, 4, 11, 15, 17) get negative coefficients, because a higher reading means less life left, and the ones that fall (7, 12, 20, 21) get positive ones. Sensors 8 and 13 break the pattern: they rise with degradation but end up with positive coefficients, because they are strongly correlated with other sensors, and the "holding the others fixed" reading from Mechanics applies.

*Self-check: standardising the inputs changes every coefficient but not a single prediction. Why?*

### 2.2 Assumptions and residual diagnostics

**Definition.** The residual $e_i = y_i - \hat y_i$ is what the model failed to explain. OLS is well behaved when residuals have mean zero at every level of the inputs (linearity), are independent of each other, and have constant variance.

**Intuition.** If the model captured everything systematic, the residuals are shapeless noise. Any pattern left in them is information the model missed, and the shape of the pattern says what kind.

**Mechanics.** Two plots answer most questions. Residuals against fitted values, or against the true target in bins, show non-linearity: the mean residual per bin should be close to zero everywhere. Residuals of one engine plotted in cycle order show dependence: runs of same-sign residuals mean the errors are correlated in time.

**Assumptions.** You need enough rows per bin for the means and spreads to be stable. With 20,631 rows and bins of 25 cycles that is not an issue here.

**Practical application.** Notebook 03, residual section. The mean residual by true-RUL bin is far from zero, and its sign flips along the range:

| True RUL | Mean residual (true − predicted) |
|---|---|
| 0 to 25 | −10.8 |
| 25 to 50 | −21.1 |
| 50 to 75 | −17.0 |
| 75 to 100 | −5.3 |
| At the cap (125) | +13.6 |

Near failure the model predicts more life than there is, and on the healthy plateau it predicts less. The capped label is piecewise and degradation is exponential (notebooks 01 and 02), so a single flat surface cannot follow both. Being late in the 25 to 75 band is the expensive direction under PHM08 and under the scenario's costs. Independence fails too: all rows of an engine share its initial wear, so their residuals are correlated. The effective sample size is closer to 100 engines than to 20,631 rows, which is the reason behind 2.7.

*Self-check: the linear model is late by about 20 cycles in the 25 to 75 band. What does that do to its PHM08 score compared with an error of the same size in the early direction?*

### 2.3 Heteroscedasticity

**Definition.** The errors are heteroscedastic when their variance changes with the level of the inputs or the target, instead of being constant.

**Intuition.** Predicting tomorrow's weather is easier than predicting next month's. An engine 10 cycles from failure is easier to place than one that is still healthy.

**Mechanics.** OLS stays unbiased under heteroscedasticity, but two things go wrong. The textbook standard errors of the coefficients are no longer valid. And every metric averaged over rows gets dominated by the regions where the errors are largest. Remedies depend on the goal: weighted least squares, transforming the target, or evaluating only where the prediction matters.

**Assumptions.** Diagnosing it needs the residual spread per bin, which again needs enough rows per bin.

**Practical application.** Notebook 03, same residual section. The residual standard deviation grows from 13.8 in the last 25 cycles to 19.6 on the plateau. The plateau also holds 39% of all training rows (8,131 rows with the label at 125). An RMSE over all rows is therefore driven largely by healthy rows, where no maintenance decision is taken. This is one reason the evaluation moves to cut points (2.8), and it anticipates level 5: prediction intervals must widen as the predicted RUL grows.

*Self-check: why would a single interval width, say ±20 cycles, be too wide near failure and too narrow on the plateau?*

### 2.4 The bias-variance trade-off

**Definition.** For squared error, the expected prediction error at a point splits into three parts:

$$
E\big[(y - \hat f(x))^2\big] = \mathrm{Var}\big(\hat f(x)\big) + \big[\mathrm{Bias}\,\hat f(x)\big]^2 + \mathrm{Var}(\varepsilon).
$$

**Intuition.** A rigid model gives almost the same answer whatever sample it is trained on, and misses the true shape: high bias, low variance. A very flexible model follows every sample closely, including its noise, and changes a lot from one sample to the next: low bias, high variance. The last term is noise that no model can remove.

**Mechanics.** None of the three terms can be measured directly on real data. What you can measure is the cross-validated error as flexibility grows: it usually falls, reaches a minimum, and rises again. The minimum is the practical answer to the trade-off.

**Assumptions.** The clean decomposition holds for squared error. For asymmetric costs like PHM08 the idea survives, but the formula does not.

**Practical application.** Notebook 03 puts both ends side by side. The linear model sits at the high-bias end: the residual pattern in 2.2 is bias, and its error barely changes between folds or with regularization (2.5). A nearest-neighbours model on smoothed features sits at the high-variance end, and it is also the one that exploits the leak most (2.7). Levels 3 and 4 move toward flexible models, which is why the validation protocol has to be trustworthy before that move.

*Self-check: where do OLS and a 5-nearest-neighbours model sit on the trade-off, and which of the two could regularization help?*

### 2.5 Ridge and lasso

**Definition.** Penalised least squares adds a cost on the size of the coefficients:

$$
\text{ridge: } \min_\beta\; \mathrm{RSS} + \lambda \sum_j \beta_j^2, \qquad \text{lasso: } \min_\beta\; \mathrm{RSS} + \lambda \sum_j |\beta_j|.
$$

**Intuition.** Ridge shrinks every coefficient toward zero and spreads weight across correlated inputs instead of letting two of them cancel each other with large opposite coefficients. Lasso can set coefficients exactly to zero, so it also selects variables.

**Mechanics.** The penalty depends on the scale of each input, so standardise first. $\lambda$ is chosen by cross-validation, and it has to use the same engine-wise folds as the evaluation: choosing it with row-wise folds would pick a $\lambda$ tuned to the leak. Lasso produces exact zeros because the L1 constraint region has corners on the axes, and the optimum tends to land on one.

**Assumptions.** Shrinkage trades a little bias for less variance. It helps only when variance is a real problem: many inputs relative to the number of independent units, or strong collinearity.

**Practical application.** Notebook 03, regularization section. With 12 sensors and 20,631 rows, collinearity is moderate: the strongest pair is sensors 11 and 12 at |r| = 0.85, and every variance inflation factor is below 6. Cross-validation picks a tiny $\lambda$, lasso zeroes no sensor, and OLS, ridge and lasso all score 22.95. That is a result: regularization removes variance, and this model has none to remove. It starts to matter at level 3, where 12 sensors × 3 windows × 3 statistics give 108 strongly correlated features.

*Self-check: why must $\lambda$ be chosen with the same engine-wise folds used to report the error?*

### 2.6 Cross-validation and out-of-fold predictions

**Definition.** In k-fold cross-validation the data is split into k parts. Each part is predicted by a model trained on the other k − 1, and the errors are averaged. The out-of-fold (OOF) predictions are the result: every observation predicted by a model that never saw it.

**Intuition.** A series of mock exams in which every question is answered by a student who did not study that question. The average mark estimates how the procedure will do on the real exam.

**Mechanics.** ISL compares three options. A single validation split wastes data and its estimate varies a lot with the split. Leave-one-out uses almost all the data but has high variance and is expensive. k-fold with k = 5 or 10 is the usual compromise. Keeping the OOF predictions, instead of only the averaged score, lets you compute any metric afterwards (RMSE, PHM08, maintenance cost) and run diagnostics like 2.2. Tuning choices on the CV score and then reporting that same score is optimistic, because the choice was fitted to it. The fix is to keep a final test set untouched, or to nest a second CV inside the first.

**Assumptions.** Folds are exchangeable, and the split reproduces the way the model will meet new data. The second condition is the subject of 2.7.

**Practical application.** Notebook 03 builds the 5 engine-wise folds that every later model reuses, and stores the OOF predictions. The FD001 test set is not touched until the end of the project. The same OOF predictions are what the level 8 simulation replays, so each engine in the simulated fleet is scored by a model that never saw it.

*Self-check: you never train on the test set, but you compare 30 model variants on it and keep the best. Why is the reported test score now optimistic?*

### 2.7 Grouped data and leakage

**Definition.** Leakage is information available during training that would not be available when the model is used. Group leakage is the common case: rows from the same unit end up on both sides of the split.

**Intuition.** An exam whose questions are lightly reworded copies of the homework. The marks are excellent and say nothing about what the student can do with a new problem.

**Mechanics.** Split by group (`GroupKFold` with `groups=unit`), so every engine is entirely in training or entirely in validation. The rule is that the unit of the split matches the unit of generalization: the model will meet new engines, so validation must contain whole engines it has not seen. Bergmeir & Benítez study the time-series version of this problem, where neighbouring observations are dependent, and recommend blocked cross-validation, which keeps contiguous blocks together. Engines are the natural blocks here: rows inside an engine are dependent, and different engines are independent, so grouping by engine is the blocked CV of this dataset.

**Assumptions.** Engines are independent of each other, which holds in the simulation. In a real fleet, engines that share a route or a maintenance shop are not fully independent, and the grouping might have to be coarser.

**Practical application.** Notebook 03, leakage section. The size of the leak depends on the model and the features, and this is the finding to keep:

| Features | Model | Split by engine | Split by row |
|---|---|---|---|
| Raw row values | Linear, KNN, random forest | 22.9, 22.9, 21.4 | 22.7, 22.7, 21.1 |
| 30-cycle rolling means + cycle | Linear | 21.9 | 21.6 |
| 30-cycle rolling means + cycle | 5-nearest neighbours | 23.0 | 3.0 |
| 30-cycle rolling means + cycle | Random forest | 20.0 | 8.2 |

With raw values, sensor noise hides the engine a row comes from, and the leak is invisible. With smoothed features, neighbouring rows of an engine become near copies, and a flexible model predicts a validation row by finding its neighbours from the same engine in training. The linear model cannot do that. A leakage demo run only with the linear baseline would show almost nothing and suggest the split does not matter. The trap appears exactly when the project moves to window features and trees, in levels 3 and 4.

*Self-check: why can a nearest-neighbours model exploit the leak and a linear model cannot, with the same features?*

### 2.8 Evaluating like the test: cut points

**Definition.** The official evaluation makes one prediction per test engine, at the last cycle of a truncated history, and compares it with the true, uncapped remaining life. Training engines have complete histories, so the development protocol has to choose where to cut them.

**Intuition.** Practise in the format of the exam. Marks from a different format do not predict the exam mark.

**Mechanics.** An RMSE over every training row, against the capped label, and an RMSE at one cut per engine, against the true RUL, measure different things. The first is dominated by healthy rows (2.3). The second is what the test and the maintenance planner care about. Where the cuts fall matters: a cut early in life has a true RUL of up to 360 cycles, which a model trained on a label capped at 125 cannot reach. With 100 engines, a single random cut per engine is also noisy, so either take several cuts per engine or repeat the draw and average.

**Assumptions.** The cut rule should come from the use case, not from the test set. The scenario gives it: planning only needs predictions for engines within reach of the shop horizon, so cuts can be restricted to histories whose true RUL is below a planning limit, for example 150 cycles. The test file happens to be similar: true RUL from 7 to 145, median 86, with 11% above 125.

**Practical application.** Notebook 03, protocol section. For the same linear model:

| Evaluation | RMSE |
|---|---|
| All rows, capped label | 22.9 |
| One uniform random cut per engine, true RUL | 48.6 on average, 43.9 to 56.6 across 20 draws |

The second number is higher because uniform cuts include early-life points with long true RUL, and it swings by 13 cycles depending on the draw. Neither number is comparable with the clock's 35.9 on the test set. The comparison that counts is the clock and the linear model evaluated under the same rule on the same OOF cut points.

*Self-check: why is the all-rows RMSE of 22.9 not comparable with the clock's 35.9, even though both are RMSEs on FD001?*

---

## Exercise

Build the development protocol that every later model uses, and the first sensor-based baseline under it.

Notebook: `notebooks/03_linear_baseline.ipynb`

1. **Protocol.** 5 engine-wise folds on the training engines, a cut-point rule tied to the planning horizon, several cuts per engine, and OOF predictions stored. The FD001 test set is not touched: it is used once, at the end of the project.
2. **Clock, recomputed.** The Weibull3 clock from notebook 02, fitted on each training fold and evaluated at the OOF cut points. This is the baseline for development.
3. **Linear baseline.** OLS on the 12 useful sensors, with residual diagnostics (2.2, 2.3) and a check of ridge and lasso (2.5).
4. **Leakage demo.** Engine-wise against row-wise splits, with raw and smoothed features, and with a linear and a flexible model (2.7). The linear model alone is not enough to show the trap.
5. **Comparison.** Clock and linear model under the same protocol, with RMSE and PHM08, and a look at the engines where each one fails worst.
6. **Production groundwork.** Log the clock and the linear baseline as the first MLflow runs, and set up CI with the existing tests.

The notebook ends with a Decisions table: fold design, cut rule, the development baseline scores, and whether the linear model beats the clock.

## Open questions

- Should the folds be stratified by lifetime, so that long-lived engines appear in every validation fold? With 100 engines and a long right tail, a fold without them never measures the hardest cases.
- Is engine age (`cycle`) a legitimate feature? It is known at prediction time, and notebook 02 shows it carries real information, but it also makes neighbouring rows of an engine easier to match (2.7).
- Should predictions be clipped to 125 when the truth at a cut can be above it? The planning limit in 2.8 decides how often that happens.
- What exactly gets logged to MLflow for each run: the fold design and cut rule as parameters, the OOF predictions as an artifact, or both?
