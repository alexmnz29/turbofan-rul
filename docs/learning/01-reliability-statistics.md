# Level 1 — Reliability statistics

**Reading:** Ebeling, *An Introduction to Reliability and Maintainability Engineering*, ch. 2–4. Optional rigor: Meeker & Escobar, *Statistical Methods for Reliability Data*, intro + Weibull chapters.

**Topics:** time-to-event distributions (exponential, Weibull), survival and hazard functions, bathtub curve, MTBF and why it is a poor metric.

## Notes

Study sheet in DIMAP form (Definition, Intuition, Mechanics, Assumptions, Practical application), same format as [00-notebook-01-fundamentals.md](00-notebook-01-fundamentals.md). Every "practical" paragraph points to a section of `notebooks/02_weibull_lifetimes.ipynb`; the numbers quoted come from that notebook, so if you change the fit you should be able to predict which numbers move. Rewrite each entry in your own words once you have run the section.

---

### 1.1 Time to failure as a random variable: F, f, S and h

**Definition.** The lifetime T of an engine is a non-negative random variable. Four functions describe it and any one determines the other three: the CDF F(t) = P(T ≤ t), the density f(t) = F'(t), the survival (reliability) function S(t) = 1 − F(t) = P(T > t), and the hazard h(t) = f(t) / S(t).

**Intuition.** S(t) answers "what fraction of the fleet is still running at age t". h(t) answers a different question: "of the engines that reached age t, what fraction fails in the next instant". S is about the whole fleet; h is about the survivors, so h can rise while f falls.

**Mechanics.** h(t) = −d/dt log S(t), so S(t) = exp(−∫₀ᵗ h(u) du). The integral is the cumulative hazard H(t). With 100 complete lifetimes the empirical survival is simply S(t) = (# lifetimes > t) / 100, a step function that drops by 0.01 at each failure. Conditional survival, the quantity that matters for a machine that has already reached age t, is P(T > t + s | T > t) = S(t + s) / S(t).

**Assumptions.** Engines are exchangeable draws from one distribution. Section 2 of notebook 01 already showed the initial wear differs per engine, so this is a fleet-level description that ignores everything the sensors know.

**Practical application.** Section 2: the empirical survival step curve for FD001. Section 4: conditional survival from the fitted model. Under the three-parameter Weibull, an FD001 engine at age 200 has about a 34% chance of surviving 50 more cycles; at age 100, 91%. That difference is the whole reason age is a usable predictor even before looking at any sensor.

*Self-check: can h(t) exceed 1? Can F(t)?*

### 1.2 The exponential distribution and memorylessness

**Definition.** T ~ Exponential(λ) has S(t) = exp(−t/λ) and constant hazard h(t) = 1/λ. It is the only continuous distribution with the memoryless property P(T > t + s | T > t) = P(T > s).

**Intuition.** A light bulb that never ages: a 1000-hour-old bulb has exactly the same chance of surviving the next hour as a new one. Failures are external shocks, not accumulated wear.

**Mechanics.** The MLE of λ is the sample mean. Log S(t) is a straight line through the origin with slope −1/λ, which is the quickest visual test: plot log of empirical survival against t and see whether it is straight.

**Assumptions.** No wear-out and no infant mortality. This is the model behind MTBF-only reliability reporting, which is why MTBF is misleading for anything that ages.

**Practical application.** Section 3: the exponential fit to FD001 is rejected outright (KS p < 0.0001), and the log-survival plot bends downward: a sharply increasing hazard. C-MAPSS engines wear out, they are not shocked to death. This is also why "RUL = expected lifetime − age" is not a constant: under memorylessness the expected residual life would be λ at every age.

*Self-check: under an exponential model, what is the expected residual life of an engine that has run 300 cycles?*

### 1.3 The Weibull distribution: shape and scale

**Definition.** T ~ Weibull(k, λ) has S(t) = exp(−(t/λ)^k) and hazard h(t) = (k/λ)(t/λ)^(k−1). k is the shape, λ the scale (the 63.2% life: S(λ) = 1/e regardless of k).

**Intuition.** k is a dial on the type of failure. k < 1: hazard decreasing (infant mortality, defective units fail first). k = 1: exponential, no ageing. k > 1: hazard increasing (wear-out); the larger k, the more failures cluster around λ and the more the lifetime behaves like a deterministic clock plus noise. k around 3.5 already looks roughly bell-shaped.

**Mechanics.** Mean (MTTF) = λ Γ(1 + 1/k); median = λ (ln 2)^(1/k); B10 life (10% failed) = λ (−ln 0.9)^(1/k). Fit by maximum likelihood with `scipy.stats.weibull_min.fit(x, floc=0)`; the `floc=0` pins the location at zero so you get the two-parameter form. On Weibull paper, log(−log S(t)) against log t is a straight line with slope k, which is the classical graphical check.

**Assumptions.** Failures can happen from t = 0. If there is a physical minimum life (nothing fails before some threshold), the two-parameter form is forced to bend to reach zero and fits the low tail badly.

**Practical application.** Section 3: FD001 gives k ≈ 4.4, λ ≈ 225, a strongly wear-out fleet. But the fit fails the KS test (p ≈ 0.01) precisely in the low tail: the model puts 5% of failures before cycle 115 while the earliest observed failure is 128. The three-parameter Weibull (free location) lands at threshold ≈ 125, k ≈ 1.8, λ ≈ 91 and fits well; so does the lognormal. Two lessons: the shape parameter you report depends on where you place zero, and "Weibull" is a family, not one answer.

*Self-check: two fleets have the same λ but k = 1.5 and k = 6. Which has the higher B10 life, and why?*

### 1.4 Hazard function and the bathtub curve

**Definition.** The bathtub curve is the qualitative hazard shape of many populations: decreasing early (infant mortality), roughly flat in mid-life (random failures), increasing late (wear-out). No single Weibull produces a bathtub; it is a mixture or a sum of hazards from different failure mechanisms.

**Intuition.** A car fleet: some lemons die in the first month, then years of occasional random failures, then everything starts to wear at once. The maintenance strategy is different in each region: burn-in, run-to-failure, and preventive replacement respectively.

**Mechanics.** For competing failure modes with independent times T₁, T₂ the system fails at min(T₁, T₂) and the hazards add: h(t) = h₁(t) + h₂(t). Survival multiplies: S(t) = S₁(t) S₂(t). A mixture of two subpopulations (each engine has one mode) instead gives a weighted average of densities, and the observed hazard is pulled toward whichever subpopulation dominates the survivors at each age.

**Assumptions.** The hazard has to be estimated from data with enough failures at each age. With 100 lifetimes the empirical hazard is very noisy; the parametric hazard is smooth but only as good as the model.

**Practical application.** Section 4: the FD001 hazard is monotone increasing, negligible for the first hundred cycles, then rising steeply (with the two-parameter fit: 0.001 per cycle at age 100, 0.013 at 200, 0.028 at 250; the three-parameter fit is exactly zero below its threshold of 125 and then climbs faster). No infant mortality, no flat region: the simulation only models wear-out, and the sensors in notebook 01 confirmed that degradation is invisible until 100–140 cycles before failure. Section 6: FD003 adds a second fault mode (fan degradation) and the fitted shape drops from 4.4 to 2.9 with a much longer right tail (max 525 vs 362); the two-parameter fit is worse there, which is what a mixture should do to a single-Weibull model.

*Self-check: if FD003 engines each suffer only one of the two faults, is the fleet a competing-risks model or a mixture? Which of the two makes the population hazard lower than either component's?*

### 1.5 MTTF, MTBF, and why a mean is a poor summary of a lifetime

**Definition.** MTTF (mean time to failure) is E[T] for non-repairable units; MTBF (mean time between failures) is the same quantity for repairable systems. Both are single numbers that hide the shape of the distribution.

**Intuition.** "MTBF 200 cycles" is compatible with every engine dying at exactly 200 and with half dying at 50 and half at 350. A maintenance schedule built on the mean is wrong for both fleets in different ways.

**Mechanics.** For the Weibull, MTTF = λ Γ(1 + 1/k) is close to λ for large k and to λ for k = 1, but the spread around it is completely different. Quantiles carry the information the mean drops: median (half the fleet gone), B10 (when to schedule the first inspections), B1 for safety-critical parts. For right-skewed lifetimes the mean sits above the median and is dragged by the tail of survivors.

**Assumptions.** MTBF as a planning number implicitly assumes constant hazard, since only then does "one failure every MTBF cycles" hold at every age.

**Practical application.** Section 2: FD001 mean 206, median 199, B10 155 (empirical). Scheduling maintenance at the MTTF would let roughly half the fleet fail first. Section 5: the fleet-only "clock" model, which predicts RUL from age using the conditional distribution, gives the baseline that any sensor model in later notebooks must beat, and its error on the test set is the number to remember.

*Self-check: for the Weibull3 fit of FD001, write MTTF, median and B10 without running the cell, then check.*

### 1.6 Censoring and the Kaplan–Meier estimator

**Definition.** An observation is right-censored when the unit was still alive when observation stopped, so you only know T > c. Kaplan–Meier estimates S(t) from a mix of failures and censored times: at each failure time tᵢ, multiply the running survival by (1 − dᵢ/nᵢ), where dᵢ is the number of failures and nᵢ the number still at risk just before tᵢ.

**Intuition.** A censored engine is a witness that says "I lasted at least this long" and then leaves. Throwing it away wastes information; counting its exit as a death biases survival downward. KM keeps it in the at-risk count until it leaves and never counts it as a failure.

**Mechanics.** Product-limit form: Ŝ(t) = ∏_{tᵢ ≤ t} (1 − dᵢ / nᵢ). With no censoring it reduces to the empirical survival. Parametric fits with censoring use the likelihood ∏ f(tᵢ) over failures times ∏ S(cⱼ) over censored units; `scipy` does not do this directly (the `lifelines` or `reliability` packages do), which is one of the open questions below.

**Assumptions.** Censoring is non-informative: the reason observation stopped is unrelated to how close the unit was to failing. In C-MAPSS test files the cut point is chosen at random by the dataset authors, so this holds by construction. In a real fleet, engines pulled early for inspection are often those that look bad, and the assumption breaks.

**Practical application.** Section 7: the 100 test engines of FD001 are all censored. Treating their observed lengths as failures pulls S(200) from 0.46 down to 0.27; treating them correctly with KM moves it to 0.54, consistent with the train-only curve and slightly higher, because the censored units add survivors without adding deaths. The right-censored test file is the same structure as an operating fleet: every engine currently flying is a censored observation.

*Self-check: why is the KM curve with censored units never below the train-only curve at the same t in this dataset?*

### 1.7 Fitting and checking a lifetime model

**Definition.** Maximum likelihood picks the parameters that make the observed lifetimes most probable. Goodness of fit compares the fitted CDF to the empirical one: graphically with a probability (QQ) plot, numerically with the Kolmogorov–Smirnov statistic or with information criteria (AIC = 2p − 2 log L) when comparing models with different parameter counts.

**Intuition.** MLE asks "which dial setting would have generated this sample most easily". The QQ plot asks "where does it lie": a good model follows the diagonal, and deviations at one end tell you which tail is wrong.

**Mechanics.** `scipy.stats.<dist>.fit` returns (shape(s), loc, scale); fix `floc=0` unless you want a threshold model. `logpdf(x).sum()` is the log-likelihood. KS p-values from `kstest` assume the parameters were known in advance; when they were estimated from the same sample the true p-values are smaller (the test is optimistic), so use KS to rank models, not to certify one. AIC penalises the extra parameter of the three-parameter Weibull, and here it still wins by a wide margin.

**Assumptions.** Independent, identically distributed complete lifetimes. Sample size 100 is enough to distinguish exponential from Weibull, but not enough to distinguish Weibull3 from lognormal (their AICs differ by 0.1 on FD001); at that point pick on physical grounds or keep both.

**Practical application.** Section 3: the quantile table and probability plots. The exponential fails everywhere; Weibull2 fails in the low tail; Weibull3 and lognormal are indistinguishable on FD001, and Weibull3 is preferred on FD003 (AIC 1126 vs 1144). Whatever you pick, carry the assumption forward explicitly: the RUL cap of 125 in notebook 01 and the Weibull3 threshold of about 125 are the same physical fact seen from two sides, and it is worth stating that in the Decisions table.

*Self-check: the KS p-value of the Weibull2 fit is 0.013. Is the true p-value larger or smaller than that, and why?*

---

## Exercise

Fit a Weibull distribution to FD001 failure times with scipy and compare against FD003.

Notebook: `notebooks/02_weibull_lifetimes.ipynb`

Concretely, the notebook should end with answers to:

1. Which family describes FD001 lifetimes, and does it matter where zero is placed (two- vs three-parameter Weibull)?
2. Is the hazard increasing (wear-out only), and where does it start to rise?
3. What RUL error does a fleet-only "clock" model achieve on the FD001 test set (RMSE and PHM08 score)? This is the baseline for notebook 03 onward.
4. How does a second fault mode (FD003) change shape, tail and fit quality?
5. How should censored test engines enter a survival estimate?

## Open questions

- Fitting Weibull with censored observations by MLE needs a custom likelihood or a package (`lifelines`, `reliability`). Worth adding as a dependency, or write the 10-line likelihood by hand and optimise with `scipy.optimize`?
- Is FD003 better modelled as a mixture of two Weibulls (one per fault mode) than as a single Weibull3? The dataset does not label which engine has which fault, so this would be an unsupervised mixture; is the gain worth it for the project, or is it enough to note that the single-model fit degrades?
- The conditional-median RUL from the clock model is monotone in age. Sensor models will not be. How to combine them: use the clock estimate as a prior, as a feature, or only as a sanity check on predictions that exceed the fleet maximum?
