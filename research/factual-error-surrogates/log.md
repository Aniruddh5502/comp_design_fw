# Research Log: Factual Error Indication in Surrogate Models

## Question and Scope
**Question:** Can "factual" error indicators (mathematical guarantees rather than probabilistic heuristics) be implemented for surrogate models without requiring ground-truth labels (FEA) for every point?
**Scope:** Focus on Certified Bounds (Lipschitz, IBP), Conformal Prediction, PDE Residuals, and Formal Verification.
**Success Criteria:** A clear mapping of each method's "factuality," scalability, and requirement for labels.

## Hypothesis Register
| Hypothesis | Prediction | Nullifiers | Status |
| :--- | :--- | :--- | :--- |
| H1: PDE Residuals are the only "physical" factual indicator. | Residuals can bound error via stability constants. | Evidence that residuals don't bound error for non-convex/non-linear operators. | Pending |
| H2: Conformal Prediction is the only "statistical" factual indicator. | CP provides coverage guarantees regardless of data distribution. | Evidence of a method that provides hard deterministic bounds without labels. | Pending |
| H3: Formal Verification (IBP/Lipschitz) is unscalable. | Bounds for 100k points will be too conservative or too slow. | Discovery of a scalable, tight certification method. | Pending |

## Query Log
(To be populated)

## Source Log
(To be populated)
