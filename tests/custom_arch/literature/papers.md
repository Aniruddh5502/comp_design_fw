# 1. [Autoencoder-based non-intrusive model order reduction for anisotropic and multi-physics problems](https://link.springer.com/article/10.1007/s00466-026-02870-y)

## Problem Solved
High-fidelity FEA is too expensive for parametric many-query tasks such as design optimization, uncertainty quantification, and digital twins.

Classical projection-based MOR, e.g. POD-Galerkin, is intrusive and often struggles with strongly nonlinear solution manifolds.

The paper uses an autoencoder (AE) in a non-intrusive three-stage framework: (i) an unsupervised AE compresses high-dimensional FEA snapshots into a latent space; (ii) a supervised regression network maps problem parameters to latent codes; (iii) the decoder reconstructs the full solution field for new parameters.

It also proposes a force-augmented variant that jointly predicts displacement fields and Neumann-boundary reaction forces, and a multi-field architecture for coupled problems such as thermo-mechanics.

## Dataset
Output type: continuous full-field outputs — displacement, reaction forces, temperature fields — not classification labels.

Source: synthetic FEA data.

Benchmarks: heterogeneous composites, anisotropic elasticity with geometric variability, and thermo-mechanical coupling.

Size: the accessible paper information does not clearly disclose the number of snapshots; the validation covers several nonlinear benchmarks.

## Results
### Methods
Train an unsupervised AE on FEA snapshots.

Train a regression network from problem parameters to latent codes.

Use the decoder as a fast surrogate for full-field reconstruction.

Extend the framework with force augmentation for Neumann boundary conditions and a multi-field architecture for coupled physics.

### Conclusion
The framework gives accurate reconstructions on all nonlinear benchmarks while remaining fully non-intrusive.

The authors conclude that combining deep learning with dimensionality reduction can produce efficient and extensible FEA surrogates.

### Critique
- Model choice: AE is appropriate because it can capture nonlinear solution manifolds better than linear POD. The force-augmented and multi-field extensions address real engineering needs.

- Data size: not clearly disclosed in the accessible material; this is a transparency weakness. The breadth of nonlinear benchmarks partly supports data sufficiency, but the claim is not fully verifiable.

- Uncertainty: the framework is deterministic and does not include built-in UQ. This is acceptable as a scope choice, but it is not suitable for probabilistic prediction without adding GP/VAE-type components.

- Conclusion justified? The reconstruction-accuracy conclusion is supported by the benchmarks. The “extensible” claim would need more independent validation across problem classes.

- Publication category: Computational Mechanics (Springer), generally Q1 in computational mechanics.


# 2. [A surrogate model for topology optimisation of elastic structures via parametric autoencoders](https://www.sciencedirect.com/science/article/pii/S0045782525007753)
