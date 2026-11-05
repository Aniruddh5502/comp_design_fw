# Research Log: Dual-Stream Alignment for Parametric FEA Surrogates

## Question and Scope
**Question:** How to align a high-capacity "Performance Stream" (Vanilla MLP) with a constrained "Certification Stream" (SNGP-lite) in a dual-stream architecture to avoid "Cognitive Dissonance" (silent failures/false alarms) without compromising the Lipschitz properties of the certifier.
**Scope:** ML architectures for Physics-Informed ML (PIML), Reliability Analysis, and General UQ. Focus on Dual-Stream/Twin-Network/Supervisor-Actor patterns.
**Success Criteria:** Identification of specific alignment mechanisms (losses, architectures, calibration) that maintain the distance-awareness of the SN-network while ensuring it reflects the performance of the flexible network.

## Sub-questions
- [ ] Which existing "Predictor-Certifier" or "Actor-Supervisor" architectures are most similar to this dual-stream approach?
- [ ] What consistency losses or architectural constraints allow a flexible network and a constrained network to stay aligned?
- [ ] How can alignment be achieved without breaking the spectral normalization $\sigma(W) \le 1$ constraint?
- [ ] Are there post-hoc calibration methods that map UQ signals to actual error in dual-stream setups?

## Hypothesis Register
| Hypothesis | Prediction | Nullifiers | Prior Confidence | Status |
| :--- | :--- | :--- | :--- | :--- |
| **H1: Consistency-based Alignment** | A penalty on the difference between the two streams (e.g., MSE or KL) will align them without breaking SN constraints. | Evidence that consistency losses force the SN-network to deviate from $\sigma(W) \le 1$ or degrade the MLP. | Medium | Pending |
| **H2: Shared Latent Space** | A shared encoder with separate heads will provide implicit alignment while allowing head-specific constraints. | Evidence that the shared encoder becomes a bottleneck that limits the Performance stream's accuracy. | Medium | Pending |
| **H3: Error-UQ Calibration** | Post-hoc mapping (e.g., Platt scaling or Isotonic regression) is more effective than in-training alignment for "Reliability Tax" problems. | Evidence that calibration fails in OOD regions where the SN-network's distance-awareness is critical. | Medium | Pending |

## Query Log
(To be populated)

## Source Log
(To be populated)

## Experiment Log
(To be populated)

## Open Issues
- Definition of "Reliability Tax" in this specific context (presumed to be the trade-off between UQ reliability and predictive accuracy).
