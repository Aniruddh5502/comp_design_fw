# Research Log: SNGP Baseline for Flexure Design Surrogates

## Question and Scope
The objective is to establish a rigorous baseline for evaluating SNGP (Spectral-normalized Neural Gaussian Process) as a surrogate model in macro-scale flexure design (FEA), specifically focusing on distance-aware Uncertainty Quantification (UQ), computational complexity, scaling, and SOTA comparisons.

**Scope:** 
- Metrics for distance-aware UQ.
- Computational complexity and scaling laws for SNGP.
- SOTA baselines in scientific ML.
- Rigorous evaluation protocols for proposing new architectures.

## Sub-questions
- [ ] What specific metrics evaluate distance-aware UQ beyond standard regression errors?
- [ ] How is compute measured for SNGP vs others?
- [ ] What are the scaling laws for SNGP regarding $N, D, H$?
- [ ] Which architectures compete with SNGP in distance-aware UQ for SciML?
- [ ] What is the ideal protocol to prove improvement over SNGP?

## Hypothesis Register
| Hypothesis | Prediction | Nullifiers | Status |
| :--- | :--- | :--- | :--- |
| SNGP scales linearly with $N$ during inference | Inference time depends on latent dimension and a small constant, not total $N$ | Evidence that SNGP requires $O(N)$ or $O(N^2)$ operations per prediction | Pending |
| Calibration Error is the primary distance-aware metric | Most papers use Expected Calibration Error (ECE) or similar | Prevalence of AURCC or Predictive Variance correlation over ECE | Pending |

## Query Log
(To be updated)

## Source Log
(To be updated)

## Experiment Log
(To be updated)

## Open Issues
- Need to clarify "distance-aware" in the context of macro-scale flexure design (is it input distance or latent distance?).
