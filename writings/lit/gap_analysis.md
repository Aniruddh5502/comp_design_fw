# Research Report: Gap Analysis for 'When Does an Ensemble Surrogate Get It Wrong?'

## 1. Hypothesis
**The Gap Exists.** 

While literature extensively documents the use of ensemble disagreement as a proxy for uncertainty (epistemic uncertainty) and utilizes this for acquisition functions in Bayesian Optimization (BO), there is a critical lack of research that specifically quantifies the **decoupling** of this relationship when the surrogate is queried by an optimizer. 

Current literature largely assumes that ensemble disagreement $\sigma_{ens}$ remains a reliable indicator of predictive error $\epsilon$ across the design space. However, in structural and MEMS optimization, the optimizer actively seeks the boundaries of constraints and the minima of the objective function—regions where the surrogate is most likely to be extrapolated or poorly calibrated. The "Gap" is the lack of a formal quantification of how "optimization bias" (the tendency of an optimizer to find regions where the surrogate is overconfident but wrong) systematically breaks the correlation between disagreement and error.

## 2. Summary of Outcomes

### 2.1. Surrogate-Based Optimization (SBO) and Hidden Failures
Research in SBO (e.g., [Confidence-Based Design Optimization...]) recognizes that insufficient sampling leads to "less confidence" in failure probability. However, these works typically treat uncertainty as a global or local parameter to be mitigated via "conservative" optima. They do not explicitly analyze the *failure rate* of the uncertainty indicator itself when pushed by an optimizer. The "hidden failure" in existing literature is treated as a sampling problem, not as a systemic failure of the ensemble disagreement metric under adversarial (optimizer-led) querying.

### 2.2. Calibration of Ensemble Uncertainty under Optimization Bias
Ensemble methods are widely accepted as a means of UQ ([Uncertainty Quantification | IBM]). However, the calibration of these ensembles is typically validated using random test sets or cross-validation. There is a profound absence of literature discussing "adversarial calibration"—how the calibration error $\left| \text{Error} - \text{Disagreement} \right|$ evolves when the input points are chosen by a $\max(\text{Objective})$ or $\min(\text{Constraint})$ operator.

### 2.3. Disagreement-based Safety Margins
In mechanical and structural design, "safety margins" are traditionally based on deterministic factors of safety or probabilistic reliability indices ([Surrogate model uncertainty quantification for reliability...]). While some modern approaches use surrogate uncertainty to drive a "more conservative" design, they do not quantify the *breakdown* point where the safety margin (derived from disagreement) becomes illusory because the ensemble is "consistently wrong" (low disagreement, high error).

### 2.4. Evolution of Failure Rate with Training Set Size (N)
Most literature focuses on how accuracy $\epsilon$ improves as $N$ increases ([Ensemble learning of multi-kernel Kriging...]). The specific question—*at what $N$ does the correlation between disagreement and error stabilize for optimizer-chosen points?*—is not addressed. Existing studies show that performance depends on the "collection of samples," but they do not track the failure rate of the uncertainty proxy as a function of $N$ under optimization pressure.

## 3. Evidence Synthesis

| Source | Stated Fact | Inferred Conclusion / Gap |
| :--- | :--- | :--- |
| [Confidence-Based Design Optimization...](https://asmedigitalcollection.asme.org/mechanicaldesign/article/143/9/091701/1096696/Confidence-Based-Design-Optimization-for-a-More) | Proposes CBDO to find conservative optima under surrogate uncertainty. | Assumes the uncertainty measure is reliable enough to define "conservatism"; doesn't quantify when the measure itself fails. |
| [Surrogate model uncertainty... for reliability](https://www.sciencedirect.com/science/article/pii/S0951832018305611) | Lack of training data leads to less confidence in predicting probability of failure. | Confirms the risk of failure but does not explore the decoupling of disagreement and error. |
| [Bayesian Optimization with Kernel Ensembles...](https://arxiv.org/html/2609.14262) | Disagreement-based acquisition provides most of the optimization gain. | Uses disagreement to *find* points, but doesn't analyze the *error* of that disagreement when the optimizer converges. |
| [Uncertainty Quantification \| IBM](https://www.ibm.com/think/topics/uncertainty-quantification) | If models disagree, it indicates uncertainty; if they agree, it suggests confidence. | This is a general heuristic; the "blind spot" is when they agree on the *wrong* answer (common in SBO). |
| [Ensemble learning of multi-kernel Kriging...](https://www.sciencedirect.com/science/article/pii/S1474034623003142) | Performance of ensemble Kriging is dependent on the collection of samples. | Highlights sensitivity to sampling, but does not distinguish between random sampling and optimization-driven sampling. |
| [SBO review (MDPI)](https://www.mdpi.com/2073-8994/14/6/1219) | Review of surrogate-model-assisted uncertainty treatment in engineering. | Focuses on propagation and quantification, not on the breakdown of the indicators themselves. |
| [On the Bayes Inconsistency...](https://arxiv.org/pdf/2512.05931) | Evaluates disagreement discrepancy for bounding model error under covariate shift. | Close to the gap (covariate shift is similar to optimizer bias), but not applied to structural/mechanical design optimization. |
| [A Unified Surrogate Framework...](https://www.mdpi.com/2571-631X/8/1/7) | Unifies UQ, Monte Carlo, and ML for reliability. | Provides a framework for UQ, but does not investigate the specific failure mode of ensembles under optimization. |
| [Practical Bayesian Optimization...](https://openreview.net/pdf?id=WryKB50aG7) | MLP ensembles outperform GP surrogates in some scientific discovery tasks. | Focuses on which model is "better," not on when the uncertainty estimate of the "better" model becomes misleading. |
| [Ensemble Surrogate Modeling... (Stratified)](https://link.springer.com/chapter/10.1007/978-981-92-1839-4_23) | Uses stratified sampling to improve model accuracy and optimization performance. | Aims to fix the sampling, but doesn't analyze the failure of the uncertainty proxy during the optimization process. |

## 4. Falsification Analysis

**Hypothesis to Falsify**: "There is no existing literature that quantifies the breakdown of the link between ensemble disagreement and predictive error under optimizer-led sampling."

**Potential Nullifying Facts**:
- If a paper exists that explicitly plots $\text{Correlation}(\sigma_{ens}, \epsilon)$ for random samples vs. optimizer-selected samples and shows a divergence.
- If a paper provides a "failure rate" of ensemble uncertainty as a function of $N$ specifically for points on the Pareto front of a structural optimization problem.

**Result of Falsification Search**: 
Searches for "ensemble disagreement predictive error optimizer bias" and "uncertainty calibration sampling bias" yielded results on *using* disagreement for optimization or *general* UQ, but no results that *critically analyze the failure* of the disagreement-error link specifically induced by the optimizer's search trajectory. The "covariate shift" paper is the closest, but it is a theoretical ML paper, not an engineering design paper.

**Conclusion**: The hypothesis holds. The gap is confirmed.
