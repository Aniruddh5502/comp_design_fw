# Research Log: Factual Error Indication in Surrogate Models

## Question
Is it mathematically possible to provide a 'factual' (non-inferred) error indication for a surrogate model's prediction without running a ground-truth simulation (FEA) for that point, relying on simple physics rules, or using density/ensemble heuristics?

## Scope
- Domain: ML surrogate models for engineering (typically PDE-based simulations).
- Constraints: No FEA per point, no sparsity-based heuristics.
- Focus: Certified bounds, Conformal Prediction, PDE residuals, Formal Verification.

## Success Criteria
- Identification of frameworks that provide guaranteed error bounds or coverage.
- Clear classification of 'Factual' vs 'Inferred' for each method.
- Analysis of scalability and handling of boundary conditions (BCs).
- A confidence level on the existence of such a 'factual' indicator.

## Assumptions
- 'Factual' means a mathematical guarantee (bound or probability) that holds regardless of the training distribution's density, given some initial calibration or structural property.
- 'Inferred' means the error estimate depends on a proxy (distance to data, variance) that might be wrong if the model is poorly specified.

## Sub-questions
1. Do Lipschitz-based bounds or IBP provide factual error guarantees for surrogates?
2. Does Conformal Prediction provide a factual coverage guarantee without per-point FEA?
3. Can PDE residuals serve as a factual error indicator for geometry-dependent problems?
4. Is formal verification (reachability) scalable to engineering surrogate dimensions?

## Hypothesis Register
- H1: Factual error indication is only possible via post-hoc calibration (Conformal Prediction) or structural bounds (Lipschitz), but cannot be 'exact' without a ground truth.
- H2: PDE residuals provide a factual upper bound on the error via stability constants (A-priori error estimates).

## Query Log
(To be updated)

## Source Log
(To be updated)

## Experiment Log
(To be updated)
