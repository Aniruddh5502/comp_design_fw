## RESEARCH TARGET

*Questions*

*Fixed Regime*
1. In a geometry-to-property surrogate with fixed loading, does ensemble disagreement stay predictive of held-out error as the training set shrinks, and at what data size does it decouple.

**a proposed diagnostic method with validation, it's Q1.**

*A Claim Right or Wrong* 
 - Expected: The rank correlation between disagreement and error degrades gradually as data shrinks.
 - Surprising(a): It stays high even when accuracy colapses. Disagrement is then a robust warning signal.
 - Surprising(b): It decouples while accuracy is still fine. Disagreement then fails first before you notice probems in accuracy.
 
*The Interest of the research*
What changes: (a) means you can use disagreement to decide when you need more simulations. (b) means you shouldn’t trust it in the low-data regime your project may actually live in.

This is also an experiment that produces a curve with a threshold, not a yes/no, and it connects to the init/order/subset attribution, since you can ask which source dominates as data changes.

In [specific system and regime], does [specific quantity] behave as [specific expectation], and where does it stop?


"we propose a gradient-structure invariance indicator that certifies whether a surrogate has captured the correct physics, and we prove/validate it across N problems"

"Or we can know the physics from before which parameter can effect which output in which manner, and make loss function on the sensitivity of each output depending on the inputs."

While training data size changes and how does that effect the models predictions and mean prediction errors. And what are the symptoms like model disagreement and mean error Spearman Corelation. Jacobean Sensitivity of each ensemble member and how they change with training data set size compared to the baseline sensitivity calculated from the training dataset itself. 
# Work to be done

1. Nested training set and one fixed test set that includes hard region*.

2. Everything Else Identical across size.: Just run in a for loop with a single model setting. The only thing varying is the training dataset size. Same dataset Initialization values for all the 5 variants. So first initialize the first model with a weight set and save it as the initialization weight and use that for all the models. Fixing the Number of gradient set, as with different size of data a 300 design point model will see less gradient steps than a model with 1000 datapoints with same batch size. So we have to  fix the gradient steps.

3. Per size: Spearman's rank correlation($\rho$  or $r_s$)[^1] between disagreement and error, the low-disagreement/high-error quadrant check(As there are 2 variables, same as the case of Pearson Corelation it will be 2x2 size box), and the Mean Disagreement[^2] to RMSE ratio[^3], with threshold fixed before looking. About the quadrant check we would have 5 checkpoints we can show these 5 points in cartesian coordinate with x axis as error and y axis as disagreements and point radius as the checkpoint indicator. Checkpoint 5 is the biggest, and 1 incrementally smaller set of training data producing model. And after keeping all these things same we run each study twice and then compare the respective runs error difference. And consider any differences between sizes smaller or equal to the difference is negligible as they comes form kernel indeterministic behavior and rounding off errors.

4. The distance-to-training data baseline.
5. One last addition: Get a grid 4/4 and at 16 points compute sensitivity of each of the 20 ensemble member and draw a distribution for each point. save their standard deviation, mean etc.  values and do this for 5 checkpoint and see how their deviation varies with training data size and this would answer if the physics they learned from data, how much does it changes when training data size changes. And how does it behave when the prediction disagreements stop tracking errors and the Mean disagreement to RMSE ratio starts to drift(if it was at ideal at first, as we expect it to be close to 1 in the ideal case)

6. Plot the member disagreement against mean error on test set. See if there are any patterns.


#### WORK DONE:

1. **Multi Fidelity Surrogate Models**: A dominant research thrust is multi-fidelity (MF) modeling, which combines cheap low-fidelity (LF) simulations with sparse high-fidelity (HF) data.

- [Leng et al. 2024](https://ouci.dntb.gov.ua/en/works/4KqkXDg9/#1) propose a variable-fidelity deep neural network based on transfer learning (VDNN-TL) that reduces the demand for high correlation between LF and HF data. In a wave rider multidisciplinary design optimization, it improved optimization efficiency by 98.9%, increased lift-to-drag ratio by 7.86% and volume ratio by 26.2%, with performance evaluation errors below 2%.

- [Sánchez-Moreno et al. (2024)](https://ouci.dntb.gov.ua/en/works/4rGydQXl/#1) apply multi-fidelity neural networks to aero-engine nacelle optimization, combining RANS (HF) and Euler (LF) CFD. Their approach achieved a 92% reduction in computational cost while matching the optimal design space found by CFD-in-the-loop optimization, with RMSE below 5%.

- [Grassi et al. (2023)](https://arc.aiaa.org/doi/10.2514/1.J061383#1) introduce RAAL (Resource Aware Active Learning), a resource-aware active learning strategy for multifidelity optimization that balances the cost of acquiring LF and HF samples. Related 2025 work extends MF surrogates to transonic aerodynamic loads estimation using Bayesian neural networks with transfer learning and to hypersonic flow fields with multi-fidelity deep operator networks.

2. **Active Learning and Adaptive Sampling**: Rather than training surrogates on a fixed dataset, active learning selectively queries the most informative points.

- [Lee (2023)](https://searchworks.stanford.edu/articles/edsndl__edsndl.VTETD.oai.vtechworks.lib.vt.edu.10919.115969#1) develops partitioned active learning for heterogeneous design spaces, failure-averse active learning for systems with implicit constraints, and multi-output extreme spatial learning for rare events in composite fuselage assembly. Guo et al. (2024) investigate active learning for adaptive surrogate improvement in high-dimensional problems. A Pareto-guided active learning strategy has been developed for accelerating multi-objective optimization of arch dam shapes.

3. **Ensemble of Surrogates**:  Since no single surrogate model is universally best for black-box functions, ensembles combine multiple models to improve robustness

- [Zhang et al. (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0957417424002926#1) provide a comprehensive review of ensemble of surrogates in black-box-type engineering optimization, covering different ensemble types, recent advances, and applications, and identifying research gaps and trends. A pointwise ensemble surrogate based on local optimal surrogates has been shown to exhibit stronger robustness and higher accuracy across problems with varying dimensions and nonlinearities


- [Random init plus shuffling is enough.](https://arxiv.org/pdf/1612.01474) The original deep ensembles paper trained every member on the full dataset. It found random initialization together with random shuffling was sufficient to obtain good performance in practice. It also argued that bootstrapping can hurt because each network sees only about 63% unique data points.

- [Random init decorrelates members more than anything else tested](https://openreview.net/forum?id=r1xZAkrFPr) Fort et al. found that the decorrelation power of random initializations is unmatched by subspace sampling that explores only a single mode. Their explanation is that deep ensembles tend to explore diverse modes in function space

- [Bootstrapping shows no benefit](https://arxiv.org/html/2510.22063) A 2025 frequentist analysis splits the bootstrap estimator into data variability and training stochasticity, and finds training stochasticity dominant, which is what deep ensembles capture

- [Bootstrapping shows no benefit.](https://openreview.net/references/pdf?id=LxsW_4WUm4) One study reports that bootstrap ensembles of deep neural networks yield no benefit over simpler baselines.




---
[^1]: **Spearman's rank correlation** ($\rho$  or $r_s$) is a non-parametric statistical measures that evaluates the strength and direction of the monotonic relationship between two variables by assessing the association between their ranked values rather than their scores. It is an preferred alternative to *Pearson Correlation* when the data are ordinal(Is a type of categorical variable where the categories have a natural, meaningful order or ranking, but the distance or intervals between those categories are uneven, unknown or not measurable), non normally distributed data(refers to statistical distribution that do not follow a symmetrical, bell-shaped curve, meaning the mean, median, and mode generally are not equal. For example: Bi-Modal, Heavy Tails, skewness) or contain outliers, as its robust to extreme values and does not asses linearity.

[^2]:**Mean Disagreement** is the average Standard Deviation(Spread) of predictions across ensemble members over the test set is a measure of *epistemic uncertainty*(How much the models disagree with each other)

[^3]:**Mean Variance to RMSE ratio**: Means approximately 1 : Well calibrated, the ensembles spread correctly reflects the true error magnitude. Less than 1 : Under-disparsed(Over confident) Members agree too tightly compared to the actual error. The ensemble underestimates its own uncertainty. Greater than 1: Over dispersed(Underconfident) - Members disagree more than the actual errors. The Ensemble overestimates its uncertainty.
**I think this can be a loss component**

