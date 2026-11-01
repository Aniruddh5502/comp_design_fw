# Architecture Evolution: Ground-Up Redesign

*Status*: Problem Definition Phase
*Goal*: Create a surrogate architecture that provides a **factual** (not inferred) measure of predictive reliability for high-fidelity engineering simulations.

## 1. Core Problem Statement
We seek to eliminate 'hidden failures' (high-confidence, high-error predictions) by fundamentally changing how the model perceives its own knowledge boundaries.

### 1.1 Key Constraints & Requirements
- **Scalability**: The architecture must be scalable across both feature dimensions (input size) and dataset size (targeting $\ge 100,000$ points).
- **Intra-Range Distance**: The model must identify "knowledge gaps" *within* the parameter bounds. We are not looking for Out-of-Range (OOD) detection, but for "sparse zones" within the training hypercube where the combination of parameters is novel or under-represented.

## 2. Falsification of Current "Inference-Based" Approaches
We have identified that most common UQ methods rely on **inference** rather than **factual deduction**.

| Method                 | Mechanism                                          | The "Inference" Trap (Why it's not factual)                                                                                                                                                                                                                                                                                                                                    |     |
| :--------------------- | :------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| **Density Estimation** | Predicts $p(x)$ (e.g., Flows, GMM)                 | **Inference**: "There are few points here $\to$ therefore the result is likely inaccurate." This is a correlation, not a fact. A model can be accurate in a sparse region if the physics are simple.                                                                                                                                                                           |     |
| **Ensemble Variance**  | Measures $\text{Var}(\hat{y}_1, \dots, \hat{y}_n)$ | **Inference**: "The models disagree $\to$ therefore the result is unreliable." This only flags epistemic uncertainty; it does not factually prove the error is high.                                                                                                                                                                                                           |     |
| **Error Prediction**   | Trains a second model to predict $                 | $y - \hat{y}$ **Path Divergence**: This requires either huge amounts of ground-truth labels (expensive FEA) or a separate physics-based error model, which changes the problem scope. |     |

## 3. The Target: Factual Reliability
We have moved from "Inferred" indicators (Density/Variance) to **Statistical Factuality**.

**The Solution: Conformal Prediction (CP)**
Instead of predicting a single value $\hat{y}$ or a heuristic variance $\sigma^2$, the model outputs a **Prediction Interval** $C(x)$ such that:
$$P(y_{true} \in C(x)) \ge 1 - \alpha$$
where $1 - \alpha$ is the user-defined confidence level (e.g., 95%).

**Why this is Factual:**
- It does not assume the error is Gaussian.
- It does not assume "sparse data = error."
- It uses a small **Calibration Set** to find a factual quantile of error.
- If the design point is in a "sparse zone" or a complex region of the manifold, the interval width $\text{Width}(C(x))$ will increase.

**The Indicator**: The **Interval Width** is our factual measure of reliability.
- **Small Width** $\to$ High factual reliability.
- **Large Width** $\to$ Low factual reliability (Knowledge Gap).

## 4. The Proposed Architecture: SNGP-Conformal Hybrid
1. **Base Model**: SNGP-RFF (provides the point prediction $\hat{y}$ and a distance-aware non-conformity score).
2. **Wrapper**: A Conformal Prediction layer that uses the SNGP variance to weight the "non-conformity" of new points.
3. **Output**: A guaranteed interval for each of the 6 outputs (4 frequencies, max stress, max deformation).
4. **Verification**: A point is flagged as a "Hidden Failure" risk if $\text{Width}(C(x)) > \text{Threshold}$.

## 5. Decision Log
- **Decision**: Abandoned PDE Residuals.
- **Reason**: PDE residuals require coordinate-based fields; parametric surrogates have geometry in the boundaries, making residuals computationally equivalent to FEA.
- **Decision**: Adopted Conformal Prediction.
- **Reason**: Provides a distribution-free, mathematically proven coverage guarantee that scales to 100k points and handles parametric inputs.
