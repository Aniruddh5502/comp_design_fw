current assumption of gap:

1. Add a term in loos function that penalizes the model if it violates structural mechanics laws.

Review:
#### A Machine Learning Approach as a Surrogate for a Finite Element Analysis: Status of Research and Application to One Dimensional Systems
id: 1
title: [A Machine Learning Approach as a Surrogate for a Finite Element Analysis: Status of Research and Application to One Dimensional Systems](https://www.mdpi.com/1424-8220/21/5/1654)
problem: Reconstruct the full-field dynamic response of a structure form a few sensors. The goal is to estimate stresses for condition-based maintanence without running the FEA.
comment: Now why would anyone run a FEA on a machines mesh that's a total system and simulating that is troublesome. We can do vibration analysis of that machine at most. Running FEA is rediculus.
Dataset: One FE simulation of a clamped aluminum beam (0.24 m, 40 Timoshenko elements, Newmark integration). It has 40,000 time steps × 41 nodes, giving 1.64M rows. A sine-chirp (0–400 Hz) distributed load is applied.
comment: Only one simulation on whose time varying data its traied, the model only learns that beams structure. And weather it can be used on other beams with the same loading, or same beams with other loading isn't verified. Only claimed.

Critique: 
1. *The validation doesn't test generalization*. Rows from One Simulation are split randomly. At \delta{t} = 0.25ms, neighbouring time steps are almost identical, so test rows are effectively interpolations between training rows. There is only one load case, so the paper never shows it works on a new loading or not. It's own discussion says the training must cover the expected loads. But it never validates that weather it fails on different load set or different geometry configuration.
2. *The XGBoost conclusion is likely a configuration artifact*. The tuned config reads "gblinear" with max depth 3, gblinear is a linear booster, so tree deapth doesn't apply, and the model is effectively a regularized linear regression. The papers claim that "XGBoost" fails on dynamic data is also not supported. The paper also says L1 regularixation "is only used in Ridge Regression", but L1 is lasso. This is a sign of sloppy hyperparameter reporting.
3. *There is no meaningfull baseline*.The system is linear and the task is essentially spatial interpolation among 5 accelerometers, A linear regression, modal expansion/superpostion or a Kalman-type observer would be the first choice for comparison.(And any of them may likely beat ANN). Without them, the paper cannot claim ML is needed. Or ML is justified.
4. *Claims that Outrun the experiments.*. The title and abstract emphasis stress, but only acceleration is predicted. Stress is never computed or validated. Table 7’s discussion even slips and says “prediction of the stress.” The paper also says “real-time” and “computationally inexpensive”, yet it reports no inference time or speedup. For a 41-node beam, the FEA itself is trivially cheap, so the benefit is never demonstrated.
5. *The metrics are hard to interpret.*
- MAE and RMSE are reported in unspecified "units". The data was z-score normalized, so it's unclear weather errors are in SD or physical units
- No normalized Error relative to RMS is given.
- The paper says RMSE “is defined as the standard deviation of the response variable”, which is wrong.
- R² of 0.98 or more is nearly meaningless for a smooth oscillatory signal evaluated on interpolation-type splits.
- Table 7 admits poor pointwise accuracy near zero crossings, and the text then says “strong pointwise accuracy” elsewhere, which is contradictory.


#### A machine learning-driven surrogate modeling approach for the structural optimization of hybrid composite wind turbine blades
id: 2
title: [A machine learning-driven surrogate modeling approach for the structural optimization of hybrid composite wind turbine blades](https://link.springer.com/article/10.1007/s00158-026-04320-y)
Problem: Use Abaqus to generate FEA solution data for some composite materials, and taking  specific wind turbine blade as a reference and generating training data. And then training a ML model which in optimization loop achieved 15.6% weight reduction relative to the baseline factorieal design approach(This means doing Grid search using Abaqus solver and selecting the optimal one form all the simulations, not like iterative refinement or gradient desccent).

Critique: The baseline choosen for this is unrealistic as there could have been used an gradient based FEA. That would have reduced their timeline.

[Simplified Swarm Optimization for Surrogate-Assisted Reliability Design of Insulated-Gate Bipolar Transistor Power Modules Using an Open-Source Process Finite-Element Model](https://arxiv.org/abs/2610.07412)
problem: worked on bipolar transistors with a surrogate that has a problem between the optimizer and the predictor.
result:

==Adressed Case==
1. Warpage predictions hold(mean absolute error 0.5%), meaning the surrogate is reliable for that output at the optimizer-reached points
2. Ceramic-stress predictions are optimistix by up to 26% at the design-space bounds reached by optimizers.
Comment: "Optimistic" here means the surrogate *under-reported* the stress and thus the optimizer thought that these design points satisfies the constraints better. And therefore selects those.

==Not Adressed Case==
Optimizer exploits the predictor’s higher uncertainty in favourable zones — Not addressed
The paper does not frame the phenomenon in terms of uncertainty exploitation. Its language is about optimistic bias, not about the optimizer preferentially sampling regions where predictive variance is high. There is no analysis of the ensemble’s predictive variance at the optimizer-reached points, no comparison of variance there versus the test set, and no discussion of an acquisition function exploiting high-uncertainty favourable zones.
Maybe because it foes find anything worth for the constraint it has that's why.. What if there are things(favourable mean prediction from the surrogate) at the high variance design points, then does the optimizer takes advantage of that.

If it does in both of the cases. Then the problem is with the surrogate. The optimizer is just doing its job. Its the surrogate that predicts favourable outputs but with increased variance and optimizer doesn't take variance into account.

If assembeled in Spearman Corelation this member disafreement and mean error, then there would be 4 zone on the chart that needs to be discussed.

**The Optimizer exploits the design space where the model is confidently wrong**

