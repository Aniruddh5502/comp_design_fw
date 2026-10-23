---
title: Documentations for baseline implementation
description: This document documents the requirements and specifics and theories needed for me to implement the SNGP(Spectral Normal Gaussian Process) for the dataset I have already generated with almost 1800+ FEA simulations. And now we will be doing the theoreticsl basics and specifics of the implementation.
---
With SNGP we need to build Ensemble NN. Where the SNGP can measure the distance of the test input form the training dataset points. Now need to do the Error Source Decomposition for this setup.
- [Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness](https://arxiv.org/html/2006.10108v2)

As there are two kinds of Uncertainities in these NN Surrogate models.

1. Aleatoric Uncertainity:
    is inherent variability in the system. More data does not remove it. 
        -   Input Randomness: material properties, geometry tolerances, boundary loads, environmental conditions that vary from one realization to another.
        -   Intrinsic Noise: Measurement Noise in experiments, or stochastic behaviour in the Physics itself(turbulence, fatigue, microstructure)
        -   manufacturing variability across units: Manufacturing scatter, so a design is really a distribution of physical objects.

2. Epistemic Uncertainity:
    Comes from lack of knowledge. It is reducible in principle.
        -   Surrogate Approximation error: The surrogate cannot perfectly reproduce the underlying model, especially in regions with few training points.
        -   Sparse or unrepresentative training data: Gaps in the inputs space where the surrogate is essentially extrapolating.
        -   Model-form error: The simulator omits or simplifies physics, so even the true response of the high-fidelity model differs from reality.
        -   Numerical Error: Mesh discritixation, solver tolerancesm and convergences.
        -   Parameter(calibration) Uncertainity: Unknown or Poorly identified model constants.

