## Hypothesis Register

**Hypothesis 1: Distance-Aware Architectures (SNGP) are the most viable for 4-input structural design.**
- **Statement:** For a 4-input regression problem, SNGP will eliminate "deception clusters" more effectively than post-hoc calibration or Deep Ensembles alone.
- **Prediction:** SNGP will show high variance in OOD regions whereas NNEs will show low variance (overconfidence).
- **Nullifiers:** Finding that SNGP fails to provide distance-awareness in low-dimensional structural manifolds; finding that a simple distance-based penalty term added to an NNE performs better with less complexity.
- **Prior Confidence:** High. SNGP is explicitly designed for distance-awareness.

**Hypothesis 2: Hybrid NN-GP Residuals provide the best "safety net" for structural design.**
- **Statement:** A small GP trained on the residuals of an NNE provides a more reliable OOD indicator than NNE disagreement.
- **Prediction:** The GP variance will spike in OOD regions even if the NNE ensemble is confident (low disagreement).
- **Nullifiers:** Finding that the GP also becomes overconfident in OOD regions if the residuals are not representative; finding that the GP is too computationally expensive for the design loop.
- **Prior Confidence:** Medium. GPs are gold-standard for UQ but can be unstable if the residual manifold is complex.

**Hypothesis 3: Post-hoc Calibration is insufficient for OOD Deception.**
- **Statement:** Temperature Scaling and Isotonic Regression can align variance with error In-Distribution (ID), but cannot solve the overconfidence problem in OOD regions.
- **Prediction:** Calibration will improve ECE on the test set (ID) but will not resolve deception clusters in OOD regions.
- **Nullifiers:** Finding a calibration method that specifically uses OOD-aware training (e.g., synthetic OOD samples).
- **Prior Confidence:** High. Calibration typically maps *predicted* variance to *actual* error; if predicted variance is 0 (overconfident), there is no signal to scale.
