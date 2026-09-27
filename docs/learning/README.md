# Study notes (DiMAP method)

Working notes, one file per level, following the plan in the project README: levels 1 to 6 build the prediction model, level 7 the maintenance planning, level 8 the production system. Each file has three sections: **Notes** (what I learned, in my own words), **Exercise** (hands-on task on C-MAPSS, with a link to the resulting notebook), and **Open questions**.

Read the Saxena et al. (2008) paper first. It explains how the data was generated and what each subset means.

[00-notebook-01-fundamentals.md](00-notebook-01-fundamentals.md) covers the pandas, statistics, matplotlib and Python fundamentals behind the EDA notebook, one DIMAP entry per element.

| Level | Topic | Key reading |
|-------|-------|-------------|
| [1](01-reliability-statistics.md) | Reliability statistics | Ebeling ch. 2 to 4; Meeker & Escobar (intro + Weibull) |
| [2](02-regression-validation.md) | Regression & validation | ISL ch. 3, 5, 6; Bergmeir & Benítez (2012) |
| [3](03-time-series-features.md) | Time series & window features | Hyndman & Athanasopoulos ch. 2 to 3 |
| [4](04-gradient-boosting.md) | Gradient boosting | ESL ch. 10; Chen & Guestrin (2016); Molnar (SHAP) |
| [5](05-asymmetric-losses.md) | Asymmetric losses & uncertainty | Koenker & Hallock (2001); Saxena et al. (2008) |
| [6](06-sequence-models.md) | Sequence models (optional) | Heimes (2008); Zheng et al. (2017); Li et al. (2018) |
| [7](07-maintenance-optimization.md) | Maintenance planning and optimization | Jardine & Tsang (2013); Williams (2013) |
| [8](08-ml-in-production.md) | Running the model in production | Huyen (2022) ch. 7 to 9; Sculley et al. (2015); Rabanser et al. (2019) |
