# Level 5: Asymmetric losses and uncertainty

**Reading:** Koenker & Hallock (2001), *Quantile Regression*, Journal of Economic Perspectives 15(4), the whole paper. Saxena et al. (2008), *Damage propagation modeling for aircraft engine run-to-failure simulation*, PHM08, the section that defines the scoring function. For calibration: Romano, Patterson & Candès (2019), *Conformalized Quantile Regression*, NeurIPS, sections 1 to 3. Optional: Gneiting (2011), *Making and Evaluating Point Forecasts*, JASA, for why the loss decides which functional you estimate.

**Topics:** the incoherence of training with squared error and evaluating with an asymmetric cost, the pinball loss and quantile regression, quantile objectives in boosting and their failure modes, the newsvendor rule for asymmetric costs, PHM08 as a loss and its surrogates, custom objectives in LightGBM, calibration and conformal intervals, what the decision layer needs from the model.

## Notes

Study sheet in DIMAP form, same format as the earlier levels. The numbers come from a protocol check on the 40-feature LightGBM from notebook 04 (seed-0 cuts, or means over 10 draws where stated); `notebooks/05_asymmetric_cost.ipynb` has to reproduce them, and if it does not, the notebook is right. Two results frame the level. First, retraining the working model with an asymmetric loss barely moves its PHM08 score on FD001, because the model is already nearly unbiased and its score is dominated by a few large errors that no loss reweighting removes. Second, the per-prediction cost sketched in the README scenario is degenerate: it is nearly minimised by predicting zero, because the cost of a shop visit and the capacity limit are missing. Both say the same thing: the asymmetry belongs mostly in the decision, with the model supplying a calibrated distribution rather than a shifted point.

---

### 5.1 Training loss and evaluation cost

**Definition.** A model trained with squared error estimates the conditional mean of the target. PHM08 and the maintenance cost are asymmetric: a late prediction costs more than an early one of the same size. Minimising one does not minimise the other.

**Intuition.** Squared error asks "what is the typical remaining life of engines like this". The planner asks "how early should I act so that the expensive mistake is rare". Those are different numbers, and the second is below the first.

**Mechanics.** For any loss, the prediction that minimises its expectation is a functional of the predictive distribution: the mean for squared error, the median for absolute error, a quantile for the pinball loss (5.2). Gneiting's point is that the loss, not the model, decides which functional you get. You can train with one loss and evaluate with another, but then the model is answering a question nobody asked.

**Assumptions.** The incoherence matters when the model has errors to shift. A model that is already unbiased and tight gains little from being pushed early, because the push also creates early errors.

**Practical application.** Notebook 05, section 1. The working model, trained with squared error, has a mean error of −0.8 cycles, 50% of its cut-point errors late, and PHM08 214 per 100 (mean over 10 draws). Shifting every prediction by a constant and searching for the PHM08-optimal shift gives −2 cycles and a gain inside the noise. The model is where squared error put it, and PHM08 does not want it much further. The level's work is elsewhere: in the tails (5.5) and in the distribution (5.7).

*Self-check: the loss decides which functional of the predictive distribution the model estimates. Which functional does PHM08 ask for, and why is it not simply "a lower quantile"?*

### 5.2 The pinball loss and quantile regression

**Definition.** For a target quantile level $\alpha$, the pinball (check) loss of a prediction $q$ against an outcome $y$ is

$$
\rho_\alpha(y - q) =
\begin{cases}
\alpha\,(y - q) & y \ge q \\[4pt]
(1 - \alpha)\,(q - y) & y < q
\end{cases}
$$

Minimising its expectation gives the $\alpha$-quantile of $y$ given the features. Quantile regression (Koenker) fits that quantile directly, without assuming a distribution.

**Intuition.** A tilted absolute error. With $\alpha = 0.1$, being above the outcome costs nine times more than being below, so the fit settles where only 10% of outcomes fall below it. Fitting several $\alpha$ gives a picture of the whole conditional distribution: where it is centred, how wide it is, whether it is skewed.

**Mechanics.** Linear quantile regression is a linear program; Koenker and Hallock's examples (Engel's food expenditure, birth weight) show how the slopes differ across quantiles when the spread depends on the features. In boosting, the same loss is handed to the trees as an objective (5.3). A quantile at $\alpha$ should cover: on new data, the fraction of outcomes below the prediction should be about $\alpha$.

**Assumptions.** Enough data in the tails. Low and high quantiles are estimated from the few observations that fall there, so they are noisier than the median.

**Practical application.** Notebook 05, section 2. Five quantile models of the 40-feature LightGBM (10-draw means): the 0.5 quantile scores like squared error (RMSE 12.5, PHM08 260, bias −0.1). The 0.25 quantile predicts 4.3 cycles lower on average and scores PHM08 228; the 0.1 quantile, 8.4 lower and 347; the 0.9, 9.0 higher and 1,148. The coverage column is the finding: the 0.1 quantile has 25% of outcomes below it instead of 10%, the 0.25 has 36%, the 0.9 has 74% instead of 90%. Every quantile is pulled toward the centre. The model's spread is too narrow, which is what 5.7 corrects.

*Self-check: the 0.1-quantile model has 25% of true values below it. Is its interval too wide or too narrow, and in which direction would a conformal correction move it?*

### 5.3 Quantile objectives in boosting, and how they fail

**Definition.** LightGBM accepts `objective="quantile"` with `alpha`. Each tree then fits the negative gradient of the pinball loss, which is $\alpha$ when the outcome is above the prediction and $\alpha - 1$ when below: a constant per side, like absolute error (level 4, question 10), but tilted.

**Intuition.** Every tree only learns which side each row is on and the tilt. It cannot see how far below the prediction a row is, so quantile boosting is slow to move and relies on many rounds. Separate models per $\alpha$ know nothing about each other, so nothing stops the 0.1 quantile from ending above the 0.9 quantile on some rows.

**Mechanics.** Crossing is checked by counting rows where a lower quantile exceeds a higher one, and fixed by sorting the quantiles per row or by fitting one model for the median and others for the offsets. A second failure is specific to this dataset. LightGBM starts every quantile model from the $\alpha$-quantile of the training labels (`boost_from_average`). With 39% of rows at the cap of 125, every label quantile from 0.61 up is exactly 125, and at $\alpha = 0.75$ the model never leaves that starting point: it predicts 125 for every engine, in every fold, with any learning rate, tree count or leaf size. Neighbouring values (0.74, 0.76) train normally. Setting `boost_from_average=False` fixes it.

**Assumptions.** The pathology is a property of the capped label meeting the objective's initialisation, not of the data. It shows up with other plateau fractions at other $\alpha$ values.

**Practical application.** Notebook 05, section 2. Crossing: 2% of cuts have the 0.1 quantile above the 0.9, and 9% have it above the median; sorting per row removes it. The 0.75 collapse is reported with its fix, because a pipeline that trains quantile models on a schedule has to detect it: a model whose predictions are all identical is the symptom.

*Self-check: why does a 39% plateau at 125 make the 0.75-quantile model start at exactly 125, and why can a constant-per-side gradient fail to move it from there?*

### 5.4 The newsvendor rule: which quantile the cost asks for

**Definition.** With a cost $c_u$ per unit of under-prediction and $c_o$ per unit of over-prediction, both linear, the prediction that minimises expected cost is the quantile

$$
\alpha^* = \frac{c_u}{c_u + c_o}
$$

of the predictive distribution. The newsvendor result: order the quantity at the critical ratio of the two costs.

**Intuition.** If being late costs the same as being early, predict the median. If being late costs nine times more, predict the 10% quantile: accept being early nine times out of ten to avoid the tenth.

**Mechanics.** For remaining life, under-prediction is early (the engine is removed with life left) and over-prediction is late (the engine fails). The README scenario sketches a late cost of 10 shop visits, flat, and an early cost of 1/200 of a visit per wasted cycle. Treated as a per-prediction cost, the ratio is about 0.0005: the rule says to predict near zero for every engine and act at once. That is not a quirk of the rule. It says the per-prediction cost is missing two things the real decision has: the fixed cost of the shop visit itself (acting now instead of in 50 cycles consumes a slot and a visit), and the capacity limit that makes slots scarce. Those enter at the decision level, in level 7.

**Assumptions.** Linear costs on each side. PHM08 is exponential on both sides, so its optimum is not a fixed quantile (5.5). A flat cost on one side and a per-unit cost on the other, as in the sketch, has an optimum that depends on the shape of the predictive distribution, not only on the ratio.

**Practical application.** Notebook 05, section 3. On the seed-0 cuts, the sketched per-prediction cost is 35 per 100 predictions for "predict zero always" and 4.8 for "the truth minus 10 cycles", against the quantile models' 251 to 870. The sketch rewards being early without limit. The evaluation function also counted an exact prediction as a failure (`d >= 0`); that is fixed and tested. The conclusion carried to level 7: the point prediction stays at the median or the mean, the asymmetry is applied when deciding, with the shop-visit cost and the capacity in the objective.

*Self-check: with a flat late cost of 10 and an early cost of 1/200 per cycle, why is "predict zero" nearly optimal per prediction, and what two costs would stop it from being optimal in the weekly plan?*

### 5.5 PHM08 as a loss, and its surrogates

**Definition.** PHM08 is $e^{d/10} - 1$ for a late error $d > 0$ and $e^{-d/13} - 1$ for an early one, summed over predictions. As a training loss its gradient is $e^{d/10}/10$ on the late side: it grows without bound, so a few large residuals dominate every tree and the fit diverges or oscillates. A surrogate keeps the asymmetry with a bounded gradient; the simplest is squared error with a weight $w > 1$ on the late side.

**Intuition.** PHM08 wants two things: predictions early rather than late, and no catastrophic errors. A weighted squared error gives the first. The second is a property of the tails of the predictive distribution, which reweighting does not change much.

**Mechanics.** Asymmetric squared loss: $w\,d^2$ for $d > 0$, $d^2$ otherwise, with gradient $2wd$ and Hessian $2w$ on the late side. Implemented as a custom objective (5.6). The weight moves the fitted functional from the mean toward a lower quantile: with $w = 4$ the model predicts 2.7 cycles lower on average and 41% of its errors are late instead of 50%.

**Assumptions.** The model has late errors that reweighting can trade for early ones at a profit under PHM08. If the late errors are the same engines whatever the loss, because the features do not see their degradation, reweighting only shifts where the errors land.

**Practical application.** Notebook 05, section 4. Asymmetric squared loss with late weight 2, 4 and 8 scores PHM08 210, 211 and 217 per 100 against 214 for squared error, with RMSE 12.2, 12.4 and 12.8 (10-draw means). The differences are inside the draw noise. What changes is the error mix: late errors above 20 cycles drop from 47 to 36 on the seed-0 cuts, early errors above 20 rise from 72 to 86, and the worst late error stays at about 45 cycles. The quantile models do the same trade at a worse exchange rate (0.25 quantile: PHM08 228). On FD001 the working model's PHM08 is not a bias problem but a tail problem, and the tail is the short-lived engines whose degradation shows late (notebook 04, section 5).

*Self-check: the asymmetric loss with weight 8 makes the model earlier on average and does not improve PHM08. What does that say about where the PHM08 score comes from?*

### 5.6 Custom objectives in LightGBM

**Definition.** LightGBM takes any twice-differentiable loss as a function returning, for each row, the gradient $g_i = \partial \ell / \partial \hat y_i$ and the Hessian $h_i = \partial^2 \ell / \partial \hat y_i^2$ at the current prediction. Each leaf's value is the Newton step $-\sum g_i / (\sum h_i + \lambda)$ over the rows in the leaf (level 4, 4.5).

**Intuition.** The gradient says in which direction and how strongly each row pulls the next tree; the Hessian says how far to trust that pull. A loss with a constant Hessian (squared error) takes equal-sized steps; a loss whose Hessian vanishes (absolute error, pinball) needs the leaf-renewal trick LightGBM applies internally for its built-in L1 and quantile objectives, which a custom objective does not get.

**Mechanics.** For the asymmetric squared loss, `grad = 2 * w * d`, `hess = 2 * w` with `w = w_late` where `d > 0` and 1 elsewhere. Pass the function as `objective=` to `LGBMRegressor`. Check it on a toy problem first: with $w = 1$ it must reproduce squared error exactly. A custom objective disables `boost_from_average`, so the first trees have to move the prediction from zero; more rounds or a higher learning rate may be needed.

**Assumptions.** The Hessian must be strictly positive, or leaves get undefined values. That is why PHM08 itself cannot be used: its gradient and Hessian explode on the late side, and a few rows take over every leaf.

**Practical application.** Notebook 05, section 4, with the three weights of 5.5, each checked against squared error at $w = 1$ before use. The exercise is as much about verifying the objective as about the scores: a custom loss with a sign error trains without complaint and produces a model that is confidently wrong.

*Self-check: why can a custom objective with a Hessian that is zero on one side produce leaves with absurd values, and how do the built-in L1 and quantile objectives avoid it?*

### 5.7 Calibration and conformal intervals

**Definition.** A prediction interval from the $\alpha$ and $1 - \alpha$ quantiles is calibrated when the outcome falls inside it in $1 - 2\alpha$ of cases on new engines. Conformal prediction adjusts an interval on held-out data so that this holds, with a guarantee that does not depend on the model being right.

**Intuition.** The model's quantiles say how uncertain the model thinks it is. Calibration data say how uncertain it actually is on engines it has not seen. The gap between the two is the model's overconfidence, and it can be measured and removed.

**Mechanics.** Split conformal, in the simplest form used here: inside each fold, hold out 16 training engines (the early-stopping monitor set serves), compute the residuals $y - \hat q_\alpha$ on their cut points, take the empirical $\alpha$-quantile of those residuals, and add it to every prediction of $\hat q_\alpha$ on the validation engines. Conformalized quantile regression (Romano et al.) does the same with one score for both ends of the interval. Coverage is then checked on the validation cuts, overall and by true remaining life, because an interval can cover on average and fail where it matters.

**Assumptions.** Calibration engines are exchangeable with the engines the interval will be used on. Calibration with 16 engines gives shifts that vary by fold (from −9 to 0 cycles for the lower end, 3 to 14 for the upper), so the correction is noisy; more calibration engines or pooling across folds would steady it.

**Practical application.** Notebook 05, section 5. The raw 0.1 to 0.9 interval covers 48.5% of true values instead of 80%, with a mean width of 17 cycles; on the plateau it covers 34%, because every quantile model is clipped at 125 and an engine with 140 cycles left is outside any interval by construction. After the conformal shift, coverage is 72.7% with width 28, and by true remaining life 94% in the last 25 cycles, 85% from 25 to 50, 70 to 75% from 50 to 100. Still below 80% in the middle band, where errors are largest, and the width is constant-shifted rather than scaled. The interval widens with predicted remaining life (5 cycles near failure, 25 at 75 to 100), which is the heteroscedasticity from level 2 made explicit. For the decision layer, the honest statement is: a 50-cycle prediction means somewhere between 36 and 64 with about 70% confidence, and no interval covers the cuts above the cap.

*Self-check: the raw interval covers 48% instead of 80%. Name two separate reasons the model's quantiles are too narrow, one about the training objective and one about the label.*

### 5.8 From a prediction to a decision

**Definition.** The decision layer (level 7) does not need a single number per engine. It needs, for each engine and each coming week, the probability that the engine fails before that week, and the cost of scheduling it then. A calibrated predictive distribution supplies the first; the scenario supplies the second.

**Intuition.** Two engines with a predicted 40 cycles are not the same decision if one is "40 ± 5" and the other "40 ± 30". The second needs a slot sooner, or more attention, and only a distribution says so.

**Mechanics.** From calibrated quantiles, $P(\text{RUL} < 7w)$ for week $w$ is read off by interpolation; from an interval, by assuming a shape between the ends. The planner then minimises expected cost under capacity, with the shop-visit cost included, which is where the asymmetry finally pays: a flat failure cost of 10 against a visit cost of 1 means acting when $P(\text{fail before next slot})$ exceeds roughly 1/11, not when the point prediction crosses a threshold. This is the newsvendor rule again, with the right costs.

**Assumptions.** The distribution is calibrated where decisions are made, the last 100 cycles. The checks in 5.7 are by remaining-life band for that reason.

**Practical application.** Notebook 05, section 6 and the Decisions table. The model that goes into the pipeline is the squared-error (or median) point model from notebook 04 for the expected value, plus conformalised 0.1 and 0.9 quantile models for the spread, with the collapse check from 5.3 and crossing fixed by sorting. The asymmetric training losses are recorded as tried and not adopted on FD001, with the reason. What the planner receives per engine: predicted remaining life, a calibrated interval, the SHAP explanation from notebook 04, and the age. What it does with it is level 7.

*Self-check: why does the decision rule "act when P(fail before the next slot) > 1/11" follow from a failure cost of 10 and a visit cost of 1, and what changes if slots are scarce?*

---

## Exercise

`notebooks/05_asymmetric_cost.ipynb`, on the 40-feature LightGBM from notebook 04 under the same protocol.

1. **Where the score comes from** (5.1): PHM08 of the working model by error size and direction, and the PHM08-optimal constant shift.
2. **Quantile models** (5.2, 5.3): $\alpha$ in {0.1, 0.25, 0.5, 0.75, 0.9} with `boost_from_average=False`, coverage against nominal, crossing rate, and the 0.75 collapse reproduced and fixed.
3. **The newsvendor check** (5.4): the sketched per-prediction cost for "predict zero", "truth minus 10" and the quantile models, and the conclusion that the asymmetry belongs in the decision.
4. **Asymmetric surrogates** (5.5, 5.6): custom asymmetric squared loss with weights 2, 4, 8, verified against squared error at weight 1; PHM08, RMSE, error mix, and worst late error.
5. **Calibrated intervals** (5.7): raw against conformal 0.1 to 0.9 intervals, coverage overall and by remaining-life band, width by predicted life.
6. **What the pipeline carries** (5.8): the point model, the calibrated quantile models, the per-engine output format; logged to MLflow.

The notebook ends with a Decisions table: training loss for the point model, how the spread is produced and calibrated, the per-engine output for the planner, and what was tried and not adopted.

## Open questions

- Conformal shifts from 16 engines per fold vary a lot. Pool the calibration residuals across folds, or use the full cross-validation residuals (CV+ or jackknife+) for a steadier correction?
- Coverage stays at 70 to 75% between 50 and 100 cycles even after the shift. Would conformalized quantile regression with a scaled score, or quantile models trained with more rounds, fix the middle band?
- The label cap makes coverage zero above 125 by construction. Should the quantile models be trained on the uncapped label, with the cap applied only to the point prediction?
- Is a flat failure cost right? A failure in service may cost more the further the engine is from a base; the scenario could make the late cost depend on where the engine is in its schedule.
