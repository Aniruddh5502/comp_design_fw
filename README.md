# Investigation into the Miscalibration of Epistemic Uncertainty in Structural Surrogates

## 1. Research Thesis
**"Quantifying the Impact of Structural Priors on the Calibration of Epistemic Uncertainty in Macro-Scale Flexures."**

This research investigates whether the injection of physics-informed basis functions improves the **calibration** of a surrogate model's uncertainty, independent of its predictive accuracy. We aim to prove that such priors do not merely reduce residual error, but fundamentally align the model's confidence (variance) with its actual performance (error). The core contribution is the demonstration that these priors enable a model to be "aware" of its own ignorance—specifically, that uncertainty increases when the physical assumptions of the prior are violated.

## 2. Scientific Objectives
- **Decouple Accuracy from Calibration**: Isolate the "Trust-Gap" by comparing physics-informed models against black-box surrogates of **identical Mean Absolute Error (MAE)**.
- **Quantify Honesty via PICP/MPIW**: Use **Prediction Interval Coverage Probability (PICP)** and **Mean Prediction Interval Width (MPIW)** to measure calibration, ensuring the model is neither overconfident nor excessively conservative.
- **The Honesty Test (Singularity Analysis)**: Validate that the model's uncertainty $\uparrow$ as Physics Validity $\downarrow$ (e.g., in thick-beam regimes where Euler-Bernoulli theory fails).
- **Reduce Optimization Regret**: Implement a UQ-aware optimization loop using a formal **Acquisition Function** (e.g., Expected Improvement) to minimize total compute cost (FEA calls) while maximizing optimality.

## 3. Methodology & Rigorous Proof Path
1. **Accuracy-Matched Baseline Death-Match**: Train a physics-informed ensemble and a black-box ensemble to the same MAE. Compare their calibration metrics to prove that physics improves honesty, not just accuracy.
2. **Data-Efficiency Ratio**: Quantify the reduction in required training samples for the physics-informed model to reach the same calibration level as the black-box model.
3. **The Singularity Test**: Evaluate the model in non-Eulerian regimes. A scientific "Pass" is achieved if the predicted uncertainty increases as the physical prior becomes invalid.
4. **Regret Analysis**: Compare a standard NSGA-II against a UQ-aware optimizer, quantifying the reduction in "Regret" (distance to true global optimum) per unit of compute.

## 4. Red Flag Warnings (Scientific Constraints)
- **The Accuracy Fallacy**: A reduction in MAE is not a reduction in miscalibration. We must prove the model is "more honest," not just "more accurate."
- **The "Feature Engineering" Trap**: Basis functions must be presented as structural priors that modulate epistemic uncertainty, not just inputs to lower the loss.
- **Metric Gaming**: To prevent "gaming" the PICP by predicting infinitely wide intervals, we will report the product of PICP and MPIW (Interval Score).
