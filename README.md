> Research Plan for the flexure surrogate-optimization work.

## 1. Title
**Can I trust the process that made the Surrogate**

## 2. The Question
When the optimizer picks a design, can we still tell from the ensemble's disagreement and mean error correlation that weather the surrogate might be wrong, based on the disagreement and mean error correlation we saw in the held out test set performance as those were verified.

## 3. Setup
1. We train our surrogate on a distribution of design space form a single LHS pool of sampling, to maintain In Order Distribution dataset for train and test.
2. The optimizer(NSGA-II) searches the surrogate for the best designs for multi objective constrained optimization.
3. Those best designs are _chosen_ because the surrogate likes them based on the contraints given to it. They are often near constraint limit. They are not random or form the same distribution as the training dataset LHS pool.

## 4. What we are Testing
1. First split the dataset. 80-20. Then use smallest from 
Training data size
    1. model no 1 training size: 300
    2. model no 2 training size: 600
    3. model no 3 training size: 900
    4. model no 4 training size: 1200
    5. model no 5 training size: 1500
to train a 20 Ensemble NN Surrogate.
**Model Choice**: Ensembles are used instead of Gaussian Processes (GP) due to the computational scaling of GPs at $N > 1000$ and the need to capture highly non-linear structural responses.

**Phase A: In-Distribution (ID) Baseline & Convergence**
- Run the test on the held-out 20% set.
- Calculate $R_{test}$ using Spearman Correlation.
- **Convergence Check**: Plot $R_{test}$ vs training size (300 $\rightarrow$ 1500) across 3-5 different random seeds.
- **Operational Stability**: $R_{test}$ is considered converged when the change in the mean $\Delta \bar{R}_{test}$ between increments is smaller than the standard deviation $\sigma_{R}$ across seeds (i.e., the change is indistinguishable from the noise floor of the training process).
- This establishes the "Normal Behavior Baseline": how the warning system performs on typical designs.

**Phase B: Out-of-Distribution (OOD) Adversarial Testing**
- Use NSGA-II to find the "optimal" designs based on the surrogate.
- **Stratified Sampling**: Select **100 candidate designs** (50 top-performing, 50 diverse Pareto).
- Run these 100 designs through the true simulator (ANSYS).
- **Simulation Failure Protocol**: 
  - **Robustness Metric**: Record the failure rate as a standalone metric: $\text{Robustness} = \frac{\text{Successful Runs}}{\text{Total Opt-Selected Runs}}$.
  - **Correlation Metric**: Calculate $R_{opt}$ (Spearman) **exclusively on successful runs** to avoid rank-tie distortion caused by software crashes.
- Calculate $R_{opt}$ and 95% confidence intervals via bootstrapping.
- **OOD Quantification**: Compare mean **Mahalanobis Distance** (which accounts for feature covariance) of OOD vs ID points to quantify "true" distributional shift.

**Phase C: Behavioral Characterization & Diagnostic Audit**
- Compare the behavior of $R_{test}$ vs $R_{opt}$.
- **The Triviality Check (kNN Baseline)**: 
  - Calculate the distance to the nearest training neighbor using the Mahalanobis distance based on the global training set covariance matrix ($\Sigma_{train}$).
  - Compare $R_{ensemble}$ vs $R_{distance}$.
  - **Triviality Threshold**: If the Spearman correlation between $\text{Disagreement}_{norm}$ and $\text{Distance}_{Mahalanobis}$ is $\rho > 0.90$, the ensemble is classified as "Trivial" (acting as a distance-to-data proxy rather than learning structural uncertainty).
- **Calibration Audit**: Plot "Predicted Error" vs "Actual Error" (Reliability Diagram).
- **Blind Spot & Caution Mapping**: 
  - Filter for "Deception" (Low Disag / High Error) and "Caution" (High Disag / Low Error) points.
  - Project 4D to 2D using **UMAP**.
  - Verify clusters using raw 4D **Mahalanobis distance** coordinates using $\Sigma_{train}$.
- **Distribution Analysis**: Analyze how the distribution of the 20-member predictions changes from the test set to the optimizer set.

**Mathematical Definitions (Fixed Anchors):**
To ensure comparability between ID and OOD phases, all metrics are normalized by the range of the original LHS pool ($\text{Range}_{LHS} = y_{max} - y_{min}$):
- **Normalized Disagreement**: $\text{Disag}_{norm} = \frac{\text{std}(\text{ensemble\_predictions})}{\text{Range}_{LHS}}$
- **Absolute Normalized Error**: $\text{Error}_{norm} = \frac{|y_{true} - \hat{y}_{mean}|}{\text{Range}_{LHS}}$
- **Sample Size**: $N=100$ for OOD is selected as a balance between statistical significance for Spearman $\rho$ and the high computational cost of high-fidelity ANSYS simulations.

## 5. What we need to track
- split train dataset in 300, 600, 900, 1200, 1500 design points.
- log analysis of the trainset. and testset.
- build model architecture. And train it and establish methods/function toe extract each members output datas for each output.
- Test the model and store for each design point, store each members predictions for each of the 6 variables.


## 6. Why This Direction

Some of the old plan is already well studied, so it would not count as new.

|Topic|What already exists|What it means for us|
|:--|:--|:--|
|Disagreement vs. error|Disagreement between ensemble members has been used to estimate error. Later work says ensemble calibration can get worse as disagreement grows ([Kirsch & Gal](https://arxiv.org/abs/2202.01851)).|Mostly classification. Don't claim "disagreement tracks error" as new.|
|Ensembles under data shift|One study found ensemble uncertainty mostly comes from single-model uncertainty, so ensembles add little extra ([NeurIPS 2022 review](https://gresearch.com/news/neurips-2022-paper-review-2)).|We need a simple baseline to show disagreement adds value (Check A).|
|Ensemble vs. GPR|Deep ensembles tend to be underconfident, and post-hoc calibration fixes this. They were compared with GPR ([Yang & Lee](https://arxiv.org/abs/2303.16210)).|GPR vs. ensemble is not new. Use GPR only as a comparison.|
|Ensembles in optimization|Uncertainty from ensembles has been used inside optimization search ([Audet et al.](https://arxiv.org/abs/2107.04360)).|Don't propose a new "uncertainty-aware optimizer."|
|Optimizers exploiting surrogate errors|Surrogates often over-promise outside their training data and mislead the optimizer ([Tan et al.](https://arxiv.org/abs/2410.11502)).|Supports our idea. But it is mostly ML benchmarks, not structural design.|
|"Variance goes up with error"|Reported routinely, for example in a recent fusion surrogate ([arXiv 2608.24785](https://arxiv.org/pdf/2608.24785)).|A baseline, not a finding.|


## Heatmap of Deception
we could collect the design points that out puted high error but low variance, and project the input 4D space on a 2D plane using UMAP and color the mean error rate in colorbar where the memebers agreed but the error was high.

## Files&Folders
src/        # Ansys simulation codes
scripts/    # The scripts that runs the code from src/
plots/      # Plots
