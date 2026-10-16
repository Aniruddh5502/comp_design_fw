# Novelty Roadmap: Path to Research Contribution

To elevate this project from a framework to a novel research contribution, we will shift from "prediction" to "trustworthy prediction."

## 1. The Novelty Pillar: UQ Calibration
Instead of just reporting an ensemble mean, we will investigate the **Reliability-Error Correlation**.
- **Hypothesis**: The disagreement between ensemble members (variance) is a statistically significant proxy for the actual prediction error (MAE) in macro-scale flexures.
- **Scientific Contribution**: Establish a "Trust-Metric" where the optimizer can decide whether to trust the surrogate or trigger a high-cost FEA run based on member disagreement.
- **Metric**: Calculate Spearman and Pearson correlations between $\sigma_{ensemble}$ and $|y_{true} - \hat{y}_{mean}|$.

## 2. The Physics Pillar: Theory-Driven Feature Engineering
Avoid the "Black Box" nature of MLPs by imbuing the model with Euler-Bernoulli beam theory.
- **Approach**: Instead of raw inputs $\{x, y, z\}$, use physics-derived features:
    - Curvature proxies: $1/R$
    - Stiffness proxies: $I = (bh^3)/12$
    - Scaling laws: $x^3, x^{-1}, \sqrt{x}$
- **Contribution**: Demonstrate that physics-informed feature engineering reduces the required dataset size for the same accuracy compared to raw MLPs.

## 3. The Optimization Pillar: UQ-Aware NSGA-II
Integrate the Trust-Metric into the optimization loop.
- **Logic**: If $\sigma_{ensemble} > \text{threshold}$, the point is marked "Uncertain." The optimizer then prioritize exploration in that region or calls ANSYS to "ground" the surrogate.
- **Contribution**: A "Hybrid-fidelity" optimization framework that minimizes FEA calls while maximizing design optimality.

## 4. Summary of Novelty Claims
| Current State | Research Goal | Novelty Claim |
| :--- | :--- | :--- |
| MLP Prediction | UQ Calibration | "Quantified correlation between ensemble variance and structural error." |
| Raw Inputs | Physics Features | "Reduction of data-dependency via beam-theory feature engineering." |
| Standard NSGA-II | UQ-Aware NSGA-II | "Adaptive fidelity optimization based on surrogate trust-regions." |

## Implementation Sequence
1. **Dataset Split**: Train/Validation/Test.
2. **Ensemble Training**: Train $N$ diverse MLPs.
3. **Correlation Study**: Run the Spearman/Pearson analysis $\rightarrow$ Write Paper Section 1.
4. **Feature Augmentation**: Add Euler-beam features $\rightarrow$ Compare Accuracy $\rightarrow$ Write Paper Section 2.
5. **UQ-Loop**: Implement the trust-based optimizer $\rightarrow$ Compare Convergence $\rightarrow$ Write Paper Section 3.
