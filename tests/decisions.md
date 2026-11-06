# Test Decisions Log

## 1. SN Ablation Study
**Date:** (prior)  
**Location:** `tests/sn_ablation_study/`

### Hypothesis
Does Spectral Normalization affect the learning ability of NNs if enough compute is given, inferring some information is stored in the weight scaling?

### Result
**True.** SN negatively affects NN learning ability. Vanilla NNs learn easily and achieve very low error floor on test data.

### Findings
1. Spectral Normalization restricts the model from learning representations and restricts expressiveness for FEA/Physics simulation data.

### Decision
SN imposes a "Reliability Tax" on accuracy. Decoupled mean prediction from reliability certification.

---

## 2. SNGP In-Distribution Test on FEA Dataset
**Date:** (prior)  
**Location:** `tests/sngp_test/`

### Hypothesis
Can SNGP predict variance for FEA data and correlate this variance with absolute error with high correlation?

### Result
**False.** Correlation ~0.2 (both Spearman and Pearson). Not correlated.

### Findings
1. SNGP prediction variance doesn't align with test-set error. Correlation too low (~0.2) to reliably predict one from the other.
2. SNGP was designed for classification OOD detection (unseen classes). It fails on continuous FEA manifolds where continuous data is the norm.

### Decision
Structural distance-awareness (SNGP) eliminated as viable path for FEA surrogate reliability.

---

## 3. Ensemble Test — Vanilla NN (10-member)
**Date:** 2026-11-06  
**Location:** `tests/ensemble_test/`  
**Guide:** `tests/ensemble_test/guide.md`

### Hypothesis
Ensemble variance (spread across 10 independently trained MLPs) correlates with absolute prediction error on parametric FEA surrogates.

### Configuration
- **Architecture:** 10 × MLP (4 → 32 → 32 → 6), ReLU hidden, no output activation
- **Optimizer:** Adam, lr=1e-3
- **Epochs:** 200, early stopping patience=50
- **Batch size:** 32
- **Split:** 70/15/15 (train/val/test) from 1897 samples

### Results

| Target | Spearman ρ | Pearson r |
|--------|-----------|-----------|
| max_stress_von_mises | 0.226 | 0.596 |
| max_deformation | 0.648 | 0.396 |
| mode_1_freq | 0.476 | 0.508 |
| mode_2_freq | 0.353 | 0.318 |
| mode_3_freq | 0.405 | 0.389 |
| mode_4_freq | 0.341 | 0.382 |

- **Best val loss:** 0.000076 (MSE, normalized)
- **Training stopped:** epoch ~180

### Interpretation
- **Spearman ρ (0.23–0.65):** Moderate monotonic relationship — variance somewhat tracks error rank
- **Pearson r (0.32–0.60):** Moderate linear correlation — stronger for stress/deformation
- **Key finding:** Ensemble variance provides *some* signal but is **not a strong factual error bound** for FEA manifolds
- **Danger zone (Q3):** Low variance + high error cases exist — ensemble can be confidently wrong

### Decision
**Ensemble variance is not a reliable factual indicator** for parametric FEA surrogates. Correlation is moderate at best; "confident but wrong" predictions (quadrant 3) occur. This path is **discarded** as a primary reliability mechanism.

### Artifacts
- `tests/ensemble_test/results/summary.md` — full auto-generated summary
- `tests/ensemble_test/results/correlation_analysis.csv` — raw statistics
- `tests/ensemble_test/results/reliability_diagram_*.png` — 6 individual quadrant plots
- `tests/ensemble_test/results/training_curves.png` — loss curves
- `tests/ensemble_test/results/correlation_heatmap.png` — heatmap
- `tests/ensemble_test/results/error_distribution.png` — error histograms

---