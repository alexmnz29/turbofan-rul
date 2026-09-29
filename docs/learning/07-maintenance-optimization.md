# Level 7: Maintenance planning and optimization

**Reading:** Jardine & Tsang, *Maintenance, Replacement, and Reliability: Theory and Applications* (2nd ed., CRC Press, 2013), the chapters on preventive replacement of components. Williams, *Model Building in Mathematical Programming* (5th ed., Wiley, 2013), the chapters on integer programming models. OR-Tools documentation, mixed-integer programming guide.

**Topics:** age-based replacement and the cost-rate criterion, deriving a threshold rule from expected cost, linear versus mixed-integer programming, capacity-constrained assignment, greedy heuristics and when they fail, rolling-horizon planning, sensitivity analysis on cost ratio and capacity.

## Notes

<!-- What I learned, in my own words -->

## Exercise

Turn RUL predictions into a weekly shop plan for the README scenario (5 slots per week, failure = 10 shop visits, wasted cycle = 1/200 shop visit).

1. From the quantile predictions of level 5, estimate for each engine the probability of failing before each upcoming week, and the expected cost of scheduling it in each week.
2. Compare three policies: a per-engine threshold rule (optimal without capacity limits), a greedy heuristic (lowest predicted RUL first), and a MILP that assigns engines to weeks under the capacity constraint.
3. Report total cost, unplanned failures and wasted cycles, and how the ranking of the three policies changes with the cost ratio and the capacity.

If the MILP does not beat the greedy heuristic under realistic capacity, say so: that is a result, and it decides whether the optimizer stays in the pipeline.

Notebook: `notebooks/06_maintenance_planning.ipynb`

## Open questions

- How many weeks ahead should the plan look? A longer horizon uses more information but relies on less certain predictions.
- Should engines enter the fleet on staggered dates, so that failures do not cluster and capacity binds in a realistic way?
