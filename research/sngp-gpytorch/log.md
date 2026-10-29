# Research Log: GPyTorch Dynamic Training Inputs (SNGP)

## Question
How to implement a training loop in GPyTorch where `ExactGP` training inputs are the dynamic output of a neural network (SNGP architecture) without triggering `RuntimeError: You must train on the training inputs!`?

## Scope
- Library: GPyTorch (ExactGP and ApproximateGP).
- Architecture: Feature Extractor (NN) -> GP Head.
- Goal: Joint training via Marginal Log-Likelihood (MLL).
- Constraint: Maintain `.train()` mode for the GP head.

## Success Criteria
- A technical explanation of why the identity check fails despite manual updates.
- A verified method to satisfy GPyTorch's internal checks for dynamic inputs.
- A concrete code implementation showing the joint training loop.

## Assumptions
- The user is using `ExactGP` which requires `train_inputs` for the covariance matrix calculation in training mode.
- The `RuntimeError` is caused by the `__call__` method of `ExactGP` (or its base classes) performing an identity check between the input passed to the model and `self.train_inputs`.

## Sub-questions
1. Why does updating `gp.train_inputs = z_train` not satisfy the identity check when passing `gp.train_inputs` to `gp()`?
2. Is there a hidden state or caching mechanism in GPyTorch that triggers this error?
3. How do existing SNGP/Deep GP implementations in GPyTorch handle dynamic latent features?
4. Does `ApproximateGP` remove this restriction, and if so, is it the appropriate architectural choice for SNGP?
