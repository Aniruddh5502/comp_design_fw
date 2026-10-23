# SNGP Baseline Specification

## Objective
Implement a Spectral-normalized Neural Gaussian Process (SNGP) as a SOTA baseline for Out-of-Distribution (OOD) uncertainty quantification. This model serves as the performance floor to evaluate the existing Ensemble surrogate's ability to detect "Deception Clusters."

## Architecture Requirements

### 1. Spectral Normalization (Latent Space)
- Every linear layer in the encoder must use Spectral Normalization.
- Goal: Ensure the mapping from input space to latent space is Lipschitz continuous, preserving distances.
- This prevents the network from "collapsing" distant OOD points into training-like latent representations.

### 2. Gaussian Process (GP) Output Layer
- The final layer must be a GP-based head rather than a standard linear layer.
- Mechanism: Use a RBF or Matern kernel in the latent space to calculate the predictive mean $\mu(x)$ and variance $\sigma^2(x)$.
- Property: $\sigma(x)$ must increase monotonically as the latent representation $z(x)$ moves away from the training set centroid.

### 3. Training Protocol
- Loss Function: Negative Log Likelihood (NLL) or MSE for the mean.
- Normalization: Use the same Min-Max scaling [-1, 1] as the Ensemble surrogate to ensure fair comparison.
- Data Split: Use the existing deterministic 80/20 split.

## Evaluation Metrics (The "Floor" Metrics)

### 1. Deception Mapping
- Plot Error $|y_{true} - \mu(x)|$ vs. Predicted Uncertainty $\sigma(x)$.
- Target: Identify the number of points where Error is high but $\sigma$ is low (Deception Gap).

### 2. OOD Spearman Correlation
- Calculate the rank correlation between the predictive variance and the absolute error on OOD samples.

### 3. PICP (Prediction Interval Coverage Probability)
- Percentage of test samples where $y_{true} \in [\mu(x) - 2\sigma, \mu(x) + 2\sigma]$.

## Integration Points
- Input: `data/processed/` (Scaled features).
- Target: Natural frequencies/stiffness from the dataset.
- Output: `models/sngp_baseline.pth` and a performance report in `baseline/results.md`.

## Reference
- [Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness](https://arxiv.org/html/2006.10108v2)