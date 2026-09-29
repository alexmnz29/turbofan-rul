# Level 4: Gradient boosting

**Reading:** Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (2nd ed., free at hastie.su.domains/ElemStatLearn), ch. 10 (boosting and additive trees), and sections 9.2 (regression trees) and 15.3.2 (variable importance). Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*, KDD, sections 2 and 3. Molnar, *Interpretable Machine Learning* (free at christophm.github.io/interpretable-ml-book), chapters on permutation feature importance and SHAP. Ke et al. (2017), *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS, for what LightGBM changes.

**Topics:** regression trees, boosting as sequential residual fitting, learning rate and number of trees, early stopping, regularization knobs, loss functions, feature importance and its traps, SHAP, boosting and leakage.

## Notes

Study sheet in DIMAP form, same format as the earlier levels. Second half of `notebooks/04_features_and_boosting.ipynb`. The numbers quoted come from a quick check under the notebook 03 protocol (seed-0 cuts) with the 99 features from level 3 and LightGBM 4.7; the notebook has to reproduce them. The headline: LightGBM on window features scores RMSE 11.8 and PHM08 218 per 100 predictions, against 18.8 and 643 for the linear model on the same features and 32.0 and 7,359 for the clock. That gap is far beyond the fold-to-fold noise of about 2 cycles. This is the working model of the project.

---

### 4.1 Regression trees

**Definition.** A regression tree splits the feature space into rectangles with a sequence of binary questions (is sensor 15 deviation above 0.4?) and predicts a constant in each rectangle: the mean of the training targets that fall there.

**Intuition.** A flowchart. A tree does not fit a formula; it sorts each engine into a bucket of similar training rows and reports what happened to them. That is why trees can follow any shape, including the accelerating degradation that defeated the linear model, and also why a single tree is jumpy: a small change in the data moves a split and changes whole buckets.

**Mechanics.** Splits are chosen greedily: at each node, the feature and threshold that most reduce the squared error. Growth stops at a maximum number of leaves, a maximum depth, or a minimum number of rows per leaf. A tree is piecewise constant, so it cannot extrapolate beyond the range of targets it has seen, which is harmless here because the label is capped.

**Assumptions.** Enough rows per leaf for the leaf mean to be stable. A leaf with 5 rows from one engine is memorization (level 2, question 3 of section 4).

**Practical application.** Notebook 04, section 3. On raw readings alone, LightGBM scores RMSE 23.8, the same as the linear model's 24.2. Trees fix the shape of the relationship, but a single noisy reading at the cut does not contain enough information for any shape to help. Given the window features, the same model reaches 11.8. Read those two numbers together: the features carry the information, the trees extract it.

*Self-check: why can a tree follow the accelerating degradation curve when a linear model cannot, and why does that not help on raw readings?*

### 4.2 Boosting: fitting the residuals, one tree at a time

**Definition.** Gradient boosting builds an additive model $F_M(x) = \sum_{m=1}^{M} \nu\, h_m(x)$, where each tree $h_m$ is fitted to the negative gradient of the loss at the current predictions, and $\nu$ is the learning rate. For squared error the negative gradient is the residual, so each tree fits what the previous trees got wrong.

**Intuition.** A committee where each new member is told only about the current mistakes. The first tree captures the big picture (healthy versus degrading), the second corrects where the first was too crude, and so on. The learning rate makes every member speak softly, so no single tree dominates and the fit improves gradually.

**Mechanics.** Two parameters set the pace: the learning rate $\nu$ and the number of trees $M$. A smaller $\nu$ needs more trees for the same fit, and usually generalizes slightly better. The trees are shallow (LightGBM grows them leaf-wise, up to `num_leaves`), so each one captures a low-order interaction. Everything else (subsampling rows and columns per tree, minimum rows per leaf) is regularization, in 4.4.

**Assumptions.** The loss is differentiable, and the sequential fit does not run long enough to fit the noise. The second is checked with early stopping.

**Practical application.** Notebook 04, section 3, with `learning_rate=0.03`, `num_leaves=15`, `min_child_samples=50`, 80% row and column subsampling. With early stopping the folds stop between 345 and 1,939 trees, and the result is RMSE 11.8. With 2,000 trees and no early stopping it is 11.4. At this learning rate the model does not overfit within 2,000 trees; the bias by bin is within 4 cycles everywhere below the cap, against 16 to 17 for the linear model on the same features.

*Self-check: for squared error, the negative gradient is the residual. What is it for absolute error, and why does that change what each tree fits?*

### 4.3 Early stopping and the validation split inside training

**Definition.** Early stopping monitors the loss on a held-out set after each tree and stops when it has not improved for a given number of rounds. The number of trees is then chosen by the data rather than fixed in advance.

**Intuition.** Listening to the committee grow and calling a halt when new members stop helping. The held-out set has to be engines the committee has never seen, or it will keep hearing improvements that are memorization.

**Mechanics.** Inside each protocol fold, the 80 training engines are split again: 64 to fit, 16 to monitor, split by engine. The final number of trees comes from the monitor set, and the model is then used as is (or refitted on all 80 with that number). The evaluation cuts of the fold are never used to stop.

**Assumptions.** The monitor set is large enough for its loss curve to be stable. With 16 engines the best iteration varies a lot between folds (345 to 1,939), which is the noise of a small monitor set, not a property of the model.

**Practical application.** Notebook 04, section 3. Early stopping and a fixed 2,000 trees give the same score, so it is not doing much here, and it costs 16 engines of training data per fold. It stays for two reasons: it is the guard that will matter with a larger learning rate or deeper trees, and it is the standard defence a reviewer expects to see. The monitor split by engine is the same rule as everywhere else in the project.

*Self-check: if the monitor set were a random 20% of rows instead of 20% of engines, which way would the chosen number of trees be biased, and why?*

### 4.4 Regularization knobs

**Definition.** The settings that limit how closely the trees can fit the training rows: number of leaves, minimum rows per leaf, row subsampling, column subsampling, and L1/L2 penalties on leaf values.

**Intuition.** Same trade-off as level 2, different dials. Fewer leaves and more rows per leaf make each tree cruder and more stable. Subsampling makes each tree see a different slice of the data, so their errors are less correlated and average out.

**Mechanics.** In LightGBM: `num_leaves` (complexity per tree), `min_child_samples` (rows per leaf), `subsample` with `subsample_freq` (rows per tree), `colsample_bytree` (features per tree), `reg_lambda` (L2 on leaf values). The XGBoost paper (section 2) writes the L2 penalty into the objective and shows how it changes the optimal leaf value. Tune inside the training folds, never on the evaluation cuts (level 2, section 2.5 and question 6).

**Assumptions.** The knobs interact. A deeper tree with a larger minimum leaf can behave like a shallow tree with a small one. Search a few combinations, not a grid of hundreds.

**Practical application.** Notebook 04, section 3. The model is robust to these settings on FD001: 63 leaves with 5 rows per leaf gives 12.3, 7 leaves with 200 rows per leaf gives 11.8, the default 15 and 50 gives 11.8. The differences are inside the fold-to-fold noise. The rows-per-leaf setting also controls how much the model can memorize individual engines (4.8). With 100 engines there is not much to tune; the knobs will matter more on FD004, with 248 engines and six regimes.

*Self-check: why does raising `min_child_samples` reduce both overfitting and the model's ability to exploit a row-wise leak? Are those the same mechanism?*

### 4.5 The loss function is a modelling decision

**Definition.** The loss is what each tree's gradient is computed from. Squared error (L2) fits the conditional mean, absolute error (L1) the median, Huber a compromise that behaves like L2 for small residuals and L1 for large ones, and the quantile loss fits a chosen percentile.

**Intuition.** Squared error takes large residuals seriously; that is usually called a weakness because outliers pull the fit. Here the large residuals are the engines about to fail, and the whole point is to take them seriously.

**Mechanics.** LightGBM accepts `objective="huber"`, `"l1"`, `"quantile"` with `alpha`, or a custom function returning gradient and Hessian. The XGBoost paper's second-order expansion (section 2.2) is why the Hessian is needed: each leaf value is a Newton step, $-\sum g_i / (\sum h_i + \lambda)$.

**Assumptions.** The evaluation metric and the training loss should agree on what matters. PHM08 penalizes late predictions more and grows exponentially with error size; squared error does neither, and Huber does the opposite of what PHM08 wants on the big errors.

**Practical application.** Notebook 04, section 3, one cell: `objective="huber"` scores 17.0 against 11.8 for squared error, and its PHM08 more than doubles. The big residuals Huber down-weights are exactly the informative ones. That is the opening of level 5, where the loss is chosen to match the cost, with quantile objectives and the scenario's asymmetric costs.

*Self-check: Huber is usually recommended as "robust". Why is robustness to large residuals harmful in this problem?*

### 4.6 Feature importance and its traps

**Definition.** Gain importance sums, for each feature, the loss reduction of every split that uses it, over all trees. Permutation importance measures how much the validation score worsens when one feature's values are shuffled, breaking its link to the target.

**Intuition.** Gain says what the model used while it was being built. Permutation says what the model depends on to predict new data. They disagree when features are redundant: the model builds on whichever of two near-copies came first (high gain, low permutation, because the other copy takes over when it is shuffled) and depends on features it used sparingly but that no other feature can replace.

**Mechanics.** Gain: `booster_.feature_importance("gain")`, normalized to a share. Permutation: `sklearn.inspection.permutation_importance` on the validation cuts of a fold, with several repeats. Molnar's caveats: permutation importance on training data measures memorization; with correlated features, shuffling one creates impossible combinations, and the importance can be split or double-counted.

**Assumptions.** Neither says whether a feature is causal, and both are unstable across folds when features are redundant. Report on validation data, across folds, and treat rankings as approximate.

**Practical application.** Notebook 04, section 4. The two rankings disagree exactly as predicted. By gain, the deviation features dominate: sensor 15 alone carries 36% and the top five deviations 76%; age has 2%. By permutation, age is the single most important feature (shuffling it raises the fold's RMSE by 6.8 cycles), followed by the absolute deviations of sensors 9 and 14 (3.5 and 3.0), which have almost no gain. Age is used in few splits but nothing else can replace it, which is the level 3 finding that age and sensors are complementary, now visible inside the model. And 62 of the 99 features have permutation importance near zero: the set is one-third signal, two-thirds copies.

*Self-check: sensor 9's absolute deviation has almost no gain but high permutation importance. What does that combination tell you about it, and about the features that have the opposite pattern?*

### 4.7 SHAP: attributing one prediction

**Definition.** SHAP values (Shapley additive explanations) distribute a single prediction's difference from the average prediction among the features, so that the contributions add up exactly to that difference. For tree models, TreeSHAP computes them exactly and fast.

**Intuition.** Importance rankings answer "what does the model rely on in general". SHAP answers "why did the model predict 40 cycles for this engine today": sensor 15 pushed it down by 30, age by 15, sensor 9 pulled it up by 5. That is the question a maintenance planner asks, and the one the level 7 pipeline should be able to answer for every engine it schedules.

**Mechanics.** `shap.TreeExplainer(model).shap_values(X)` gives one value per feature per row. Averaging absolute values over rows gives a global importance that is consistent (a feature that matters more never gets a lower value), unlike gain. Dependence plots (SHAP value against feature value) show the shape the model learned: for a deviation feature, the point where remaining life starts to fall. `shap` is a project dependency.

**Assumptions.** SHAP explains the model, not the engine: if the model relies on a proxy, SHAP will faithfully report the proxy. With correlated features the credit is split between copies, as with every other attribution.

**Practical application.** Notebook 04, section 4. Global SHAP importance on validation cuts, compared with the gain and permutation rankings; a dependence plot for sensor 15 deviation and for age; and the explanation of two engines: one the clock got wrong and one the linear model got wrong (notebook 03, section 5). The engine-level explanation is a candidate output of the weekly plan in level 7.

*Self-check: SHAP values for a prediction add up to the prediction minus the average prediction. Why does that additive property make them usable in a maintenance report where gain importance is not?*

### 4.8 Boosting and leakage

**Definition.** A boosted tree ensemble can memorize individual training rows when leaves are small and trees are many. With a row-wise split, that memorization scores as skill.

**Intuition.** Level 2 measured the leak for nearest neighbours (25.9 to 6.4) and a random forest (23.9 to 11.5) on smoothed features. Boosting sits with the forest: it splits space into leaves rather than looking up neighbours, and `min_child_samples` sets how small a leaf can get.

**Mechanics.** Same check as notebook 03 section 4: train with row-wise folds, score on the same cuts. The gap between engine-wise and row-wise scores is the leak the model can exploit, and it grows with `num_leaves` and shrinks with `min_child_samples`.

**Assumptions.** The engine-wise score is the one that predicts performance on new engines. The row-wise score has no use except as a warning.

**Practical application.** Notebook 04, section 3. LightGBM on all features moves from 11.8 (by engine) to 8.5 (by row). Smaller than the nearest-neighbour leak, but a reviewer who saw 8.5 reported would assume a split-by-row mistake, and would be right to check. The engine-wise protocol is what makes 11.8 a defensible number.

*Self-check: the leak for LightGBM is 3.3 cycles, for KNN 19.5. What about the two models explains the difference, and which LightGBM setting would make its leak larger?*

---

## Exercise

Second half of `notebooks/04_features_and_boosting.ipynb`. The trained model and its feature list are what level 5 retrains with a different loss and what level 7 schedules from.

1. **LightGBM under the protocol** (4.2, 4.3): early stopping on an engine-wise monitor split inside each fold, 30-draw mean and per-fold scores, against the linear model on the same features and the clock. Expected RMSE about 11.8, PHM08 about 220 per 100.
2. **Knobs and loss** (4.4, 4.5): three settings of leaves and rows per leaf, and one run with the Huber loss, in one table. Show that the knobs barely matter on FD001 and the loss matters a lot.
3. **Leakage check** (4.8): row-wise split on the same cuts, one row in the table.
4. **Importance three ways** (4.6, 4.7): gain, permutation on validation cuts, and SHAP, with the disagreements explained. Prune the feature set to the features that carry signal and check the score holds.
5. **Two engines explained** (4.7): SHAP for one engine the clock got wrong and one the linear model got wrong.
6. **Log and promote**: the model logged to MLflow with its feature list and per-fold scores; feature code in `features.py`, model configuration in `config.py`.

The notebook ends with a Decisions table: feature set, model settings, the working model's development scores, and what level 5 changes.

## Open questions

- Early stopping does nothing on FD001 at this learning rate, and costs 16 training engines per fold. Keep it as a guard, or refit on all 80 engines with the chosen number of trees?
- The pruned feature set (about a third of the 99) is what the pipeline should carry. Should pruning be done once by hand from the importance tables, or automatically inside the training folds so it is reproducible?
- SHAP values on validation cuts explain the model's behaviour on new engines. For the weekly plan, should the explanation be per engine (why this prediction) or per decision (why this engine is scheduled this week and not next)?
- The model is unbiased below the cap but 20 cycles early above it, by construction of the label. Level 5 changes the loss; does it also need to revisit the cap?
