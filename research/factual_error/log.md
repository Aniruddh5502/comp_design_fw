# Research Log: Factual Error Indication in Surrogate Models

## Question
Is it possible to provide a "factual" (non-inferred) error indicator for surrogate model predictions without requiring FEA ground truth for every point?

## Scope
- Domain: ML Surrogate Models for Engineering (PDE-based, e.g., Flexures).
- Scale: Up to 100k points.
- Focus: Certified Bounds, Conformal Prediction, PDE Residuals, Formal Verification.

## Success Criteria
For each method, a determination on:
1. Factual vs. Inferred status.
2. Scalability to 100k points.
3. Handling of geometry/BCs.
4. Evidence grade (A-D).

## Hypothesis Register
- H1: A Posteriori PDE residuals provide a certified upper bound on error if the stability constant is known. (Prior: Medium)
- H2: Conformal Prediction provides a factual marginal coverage guarantee regardless of model architecture. (Prior: High)
- H3: Formal verification (IBP/Reachability) is factual but fails to scale to 100k points due to the wrapping effect. (Prior: High)
