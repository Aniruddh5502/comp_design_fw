# Research Log: Mitigating Proxy UQ in NN Ensembles for Structural Design

## Question and Scope
**Question:** Which techniques most effectively mitigate the "Proxy UQ" problem (specifically overconfidence/deception clusters in OOD regions) for NN Ensembles used as surrogate models in a 4-input structural design problem?

**Scope:**
- Focus: Neural Network Ensembles (NNE) and their uncertainty quantification (UQ).
- Problem: Disagreement as a proxy for uncertainty $\rightarrow$ OOD failure $\rightarrow$ Deception clusters.
- Inputs: Small input dimensionality (4 inputs).
- Target: Structural design (implies regression, smooth but potentially non-linear manifolds).

**Success Criteria:**
- A justified recommendation of a mitigation strategy (or combination).
- Analysis of Calibration, Architectures, OOD Regularization, and Hybrid models.
- Evidence-backed confidence levels for each technique.

**Assumptions to Test:**
- Assumption 1: 4-input problems are "low-dimensional," making distance-based methods (Mahalanobis) more viable than in high-dimensional image data.
- Assumption 2: The "Deception" cluster is primarily caused by the lack of a principled prior in NNEs.

## Sub-questions
- [ ] How do standard calibration methods (Temperature Scaling, etc.) perform specifically for *regression* variance in NNEs?
- [ ] Do SNAPs or EDL provide a principled way to handle OOD uncertainty compared to simple NNE disagreement?
- [ ] Does combining MC-Dropout with Deep Ensembles provide additive benefits for OOD detection?
- [ ] How effective are distance-aware terms for low-dimensional structural design inputs?
- [ ] Does a GP-residual layer effectively "catch" the NNE's OOD failures?

## Hypothesis Register
(To be populated in Step 3)

## Query Log
(To be populated)

## Source Log
(To be populated)

## Experiment Log
(To be populated)

## Open Issues
- Need to find benchmarks specifically for *regression* UQ, as many UQ papers focus on classification.
