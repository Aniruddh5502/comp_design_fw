## RESEARCH TARGET

## CLAIM

**In an ANSYS-verified MEMS case study, surrogate error at optimizer-selected Pareto designs is larger than held-out error. An ensemble with calibration near the Pareto region gives honest error bounds there, and a GPR baseline does not do better.**

## STERRING DIRECTION

1. The model matches ANSYS everywhere.
This is the best case for a "does it work" paper. The story is that your framework gives trustworthy designs, and you have solid proof across many designs, not just one. Lean into that evidence.

2. The chosen designs are noticeably worse than the model claimed.
This is more interesting. The story is that the optimizer finds the model's weak spots, and you can measure how much. Then show that your 20-model ensemble's disagreement flags the bad designs ahead of time. That becomes the paper: how to avoid being fooled by your own surrogate.

3. Some outputs are fine and others are off.
For example, frequencies match but stress doesn't. The story is which outputs you can trust and which need a safety margin. A nice extra is to add your verified designs to the training data, retrain once, and show the error shrinks.

4. The simpler GPR model does as well or better.
With only four inputs, this is quite possible. Report it honestly. The story becomes practical guidance: for small design problems, a simple model is enough, and the ensemble's real advantages are its uncertainty estimate and its derivatives. *That is still publishable, and it's the kind of result reviewers trust.*

## FRAMING

1. Ask q question the paper answers
"Here are my results" is weak. "When an optimizaer picks designs forma  surrogate, how far off are they from ansys, and can the ensemble warn us?" is a clear question. The findings are then the answer, and a reviewer can see why they matter.

2. Keep Facts and Opinions visibly seperate.
> *RESULTS*: Only what the numbers show, with no spin
> *DISCUSSION*: "We interpret this as..." -> This is the opinion part
> *LIMITATIONS*: one device, four inputs, one solver, one mesh setting. State them yourself before a reviewer does.

*An opinion backed by a number is respected. An opinion without one gets challenged.*

3. Evidence papers need stricter methods, not looser ones.
Because you are not offering a clever method, the only thing a reviewer can judge is whether the evidence is trustworthy. That is why the earlier safeguards matter:

Lock the list of verified designs before you run them.
Include the simple baselines (GPR and a beam-theory estimate).
Report bad results as plainly as good ones.

4. of the designs the optimizer called feasible, how many actually violate a constraint in ANSYS? That is easy to understand and hard to argue with.

#### WORK DONE:

1. **Multi Fidelity Surrogate Models**: A dominant research thrust is multi-fidelity (MF) modeling, which combines cheap low-fidelity (LF) simulations with sparse high-fidelity (HF) data.

- [Leng et al. 2024](https://ouci.dntb.gov.ua/en/works/4KqkXDg9/#1) propose a variable-fidelity deep neural network based on transfer learning (VDNN-TL) that reduces the demand for high correlation between LF and HF data. In a waverider multidisciplinary design optimization, it improved optimization efficiency by 98.9%, increased lift-to-drag ratio by 7.86% and volume ratio by 26.2%, with performance evaluation errors below 2%.

- [Sánchez-Moreno et al. (2024)](https://ouci.dntb.gov.ua/en/works/4rGydQXl/#1) apply multi-fidelity neural networks to aero-engine nacelle optimization, combining RANS (HF) and Euler (LF) CFD. Their approach achieved a 92% reduction in computational cost while matching the optimal design space found by CFD-in-the-loop optimization, with RMSE below 5%.

- [Grassi et al. (2023)](https://arc.aiaa.org/doi/10.2514/1.J061383#1) introduce RAAL (Resource Aware Active Learning), a resource-aware active learning strategy for multifidelity optimization that balances the cost of acquiring LF and HF samples. Related 2025 work extends MF surrogates to transonic aerodynamic loads estimation using Bayesian neural networks with transfer learning and to hypersonic flow fields with multi-fidelity deep operator networks.

2. **Active Learning and Adaptive Sampling**: Rather than training surrogates on a fixed dataset, active learning selectively queries the most informative points.

- [Lee (2023)](https://searchworks.stanford.edu/articles/edsndl__edsndl.VTETD.oai.vtechworks.lib.vt.edu.10919.115969#1) develops partitioned active learning for heterogeneous design spaces, failure-averse active learning for systems with implicit constraints, and multi-output extreme spatial learning for rare events in composite fuselage assembly. Guo et al. (2024) investigate active learning for adaptive surrogate improvement in high-dimensional problems. A Pareto-guided active learning strategy has been developed for accelerating multi-objective optimization of arch dam shapes.

3. **Ensemble of Surrogates**:  Since no single surrogate model is universally best for black-box functions, ensembles combine multiple models to improve robustness

- [Zhang et al. (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0957417424002926#1) provide a comprehensive review of ensemble of surrogates in black-box-type engineering optimization, covering different ensemble types, recent advances, and applications, and identifying research gaps and trends. A pointwise ensemble surrogate based on local optimal surrogates has been shown to exhibit stronger robustness and higher accuracy across problems with varying dimensions and nonlinearities



