# Research Strategy: Surpassing SOTA in Ensemble-based Structural Surrogates

## 1. Objective
The overarching goal is to discover and validate a surrogate architecture that outperforms current state-of-the-art (SOTA) methods in terms of OOD uncertainty quantification. Specifically, we aim to eliminate "Deception Clusters" (low variance, high error) more effectively than current distance-aware models.

Note: "Out-of-Distribution" (OOD) in this context refers to distributional shift—points that reside in low-density regions of the original Latin Hypercube Sampling (LHS) distribution—even if they remain within the physical parameter bounds.

## 2. The Baseline Requirement
To justify any claim of improvement, a deterministic performance floor must be established. We cannot evaluate a new hypothesis without a calibrated baseline.

### 2.1 SOTA Baseline: SNGP
The current gold standard for distance-aware OOD uncertainty in neural networks is **Spectral-normalized Neural Gaussian Processes (SNGP)**. 
- **Mechanism**: Uses spectral normalization to preserve distances in the latent space and a GP output layer to ensure variance increases as a function of distance from training data.
- **Purpose in this Project**: SNGP serves as the "Target to Beat."

## 3. Research Roadmap

### Phase 1: Establishing the Floor (Baseline)
1. **Implementation**: Build the SNGP architecture.
2. **Calibration**: Evaluate $\sigma(x)$ vs $|y_{true} - \mu(x)|$ on a held-out OOD test set.
3. **Metrics**: Document the Spearman Correlation and PICP (Prediction Interval Coverage Probability).
4. **Result**: This establishes the current SOTA performance level for our specific structural design manifold.

### Phase 2: Hypothesis Building & Innovation
Once the baseline is fixed, we will hypothesize novel improvements. Potential directions include:
- Improving the latent space mapping beyond Spectral Normalization.
- Hybridizing SNGP with specific physics-informed constraints.
- Developing more efficient distance-aware kernels for high-dimensional structural data.

### Phase 3: Validation & Comparison
Any new hypothesis will be tested against the SNGP baseline. A method is considered successful only if it shows a statistically significant improvement in PICP or a reduction in the "Deception Gap" (the number of overconfident OOD points).

## 4. Evaluation Framework
To maintain research rigor, every comparison will be measured using:
- **Deception Map**: Scatter plot of `Error vs. Variance`.
- **OOD Spearman**: Rank correlation between uncertainty and error.
- **PICP**: The fraction of true values captured within the predicted $2\sigma$ uncertainty interval.

## 5. References
- [Uncertainty-aware Deep Learning with SNGP (TensorFlow)](https://www.tensorflow.org/tutorials/understanding/sngp)
- [Deep Ensembles Work, But Are They Necessary? (arXiv)](https://arxiv.org/pdf/2202.06985)
