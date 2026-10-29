# Research Log: GPyTorch SNGP RuntimeError Fix

## Problem Statement
User is getting `RuntimeError: You must train on the training inputs!` in GPyTorch when implementing SNGP. This is caused by the `ExactGP` requirement that `gp(x)` must have `x is gp.train_inputs` during training. Since SNGP has a dynamic `FeatureExtractor`, `train_inputs` change every epoch, leading to identity check failures.

## Sub-questions
- Why does GPyTorch enforce this identity check and how can it be bypassed without `eval()`?
- How do existing SNGP/Deep GP implementations handle dynamic latent features?
- Is `ApproximateGP` the standard way to avoid the `train_inputs` constraint?
- What is the correct training loop structure for joint FeatureExtractor + GPHead optimization?

## Hypothesis Register
| Hypothesis | Prediction | Nullifiers | Status |
| :--- | :--- | :--- | :--- |
| ExactGP is fundamentally unsuitable for end-to-end training with dynamic features | User will find that all SNGP/Deep GP impls use Variational/Approximate GPs | Evidence of a successful ExactGP SNGP implementation using a specific trick | Pending |
| Updating `train_inputs` requires a specific GPyTorch internal state reset | Using a specific method to update inputs will satisfy the check | Identity check still fails after using the recommended update method | Pending |

## Query Log
(To be updated)

## Source Log
(To be updated)

## Open Issues
- Check if `set_train_data` or similar methods exist.
- Confirm if `ApproximateGP` is the "SOTA" way for SNGP.
