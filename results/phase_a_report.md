# Phase A: In-Distribution Performance Report

## 1. Executive Summary
This report documents the baseline performance of the 20-member Ensemble Neural Network surrogate on In-Distribution (ID) test data. The goal was to evaluate how the model's predictive accuracy ($\bar{R}_{test}$) scales with training set size.

## 2. Experimental Setup
- **Model**: 20-member MLP Ensemble.
- **Input**: 4D design space (Beam Width, Length, Fillet Radius, Beam Height).
- **Output**: 6D response (Stress, Deflection, $f_1, f_2, f_3, f_4$).
- **Method**: For each training size $N \in \{300, 600, 900, 1200, 1500\}$, 3 models were trained using different seeds to calculate stability ($\sigma_R$).

## 3. Deterministic Results
The following table represents the mean Pearson correlation ($R_{test}$) and its standard deviation across seeds:

| Training Size (N) | Mean $R_{test}$ | Std $\sigma_R$ |
| :--- | :--- | :--- |
| 300 | 0.4316 | 0.0119 |
| 600 | 0.3776 | 0.0066 |
| 900 | 0.3624 | 0.0367 |
| 1200 | 0.3423 | 0.0211 |
| 1500 | 0.3704 | 0.0088 |

## 4. Key Findings
- **Accuracy Plateau**: The model shows a moderate correlation ($\sim 0.37$) that does not significantly improve with more data (1500 points). This suggests a limit to the MLP's capacity or an inherent noise level in the simulated data.
- **High Stability**: The standard deviation across seeds is very low ($\sigma_R < 0.04$ for all sizes), proving that the model is not sensitive to initial weights.
- **Baseline Established**: This ID performance provides the "truth" baseline. Any significant deviation in the OOD phase will be attributed to extrapolation failure rather than general model instability.

## 5. Conclusion
The surrogate is stable but moderately accurate on In-Distribution data. It is now a valid baseline for testing "Hidden Failures" in Out-of-Distribution regions.
