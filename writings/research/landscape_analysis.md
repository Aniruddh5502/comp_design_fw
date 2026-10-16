# Landscape Analysis: ML Surrogate Models for Mechanical Flexures

## Hypothesis
The current research landscape for mechanical surrogate models has shifted from simple MLP regression to Physics-Informed Neural Networks (PINNs) and GNNs for full-field stress prediction. While high-fidelity models exist for complex geometries, there is a distinct gap in "lightweight, trustworthy" surrogates for macro-scale flexures that specifically link ensemble member disagreement to actual prediction error (UQ Calibration) and utilize basic beam theory for feature engineering rather than complex PDE loss functions.

## Summary
Current SOTA focuses on two extremes: 
1. **High-Complexity**: GNNs and Mesh-based models for arbitrary shapes.
2. **High-Theory**: PINNs that embed PDEs into loss functions for inverse problems.

For simple macro-scale flexures, the field relies heavily on standard regression (Kriging, MLP) or ensembles. The most promising novelty for this project lies in the "Trustworthiness" aspect—quantifying the correlation between ensemble variance and mean absolute error to create a reliability metric for the optimizer.

## Sources

### 1. [SOTA for Stress/Deflection (GNNs)](https://www.researchgate.net/publication/383602186_Fast_Prediction_of_Stress_Distribution_A_GNN-based_surrogate_model_for_unstructured_mesh_FEA)
- States that GNNs are SOTA for unstructured mesh FEA and full-field stress prediction.
- Inference: For simple parametric beams, GNNs are overkill; however, their success validates that structural data is highly learnable.

### 2. [PINNs for Beams](https://repository.tudelft.nl/record/uuid:b0dac776-9c30-4f97-a8da-acb9b40e9579)
- Discusses PINN-based methodologies for simulating beam systems.
- Inference: Embeds PDEs into the network. For our work, a "Physics-Informed Feature" approach (Euler-Bernoulli inputs) is a lighter, more practical alternative to full PINNs.

### 3. [Euler-Bernoulli in NN Loss](https://www.sciencedirect.com/science/article/pii/S0888327024010884)
- Uses Euler-Bernoulli beam models in the physical loss function.
- Inference: Confirms that using beam theory as a constraint/loss is a recognized scientific method for structural surrogates.

### 4. [Ensemble-based UQ (Structural Safety)](https://www.sciencedirect.com/science/article/pii/S0167473026000123)
- Introduces ensemble-based decomposition to separate irreducible randomness from reducible uncertainties.
- Inference: Validates the "Ensemble" approach for UQ. Our proposed Spearman/Pearson correlation study is a specific application of this broader UQ decomposition.

### 5. [Robust DNN Surrogates with UQ](https://onlinelibrary.wiley.com/doi/10.1002/sam.11610)
- Emphasizes that UQ is essential when simulators are computationally expensive.
- Inference: Supports the need for the `predict` and `optimization` modules to be UQ-aware.

### 6. [Mesh-Free Surrogate Comparison](https://www.mdpi.com/2076-3417/11/20/9411)
- Compares various mesh-free ML/DL methods for FEM.
- Inference: Suggests that hyperparameters and model selection are non-trivial; validates the need for the "Ensemble" to hedge against single-model failure.

### 7. [Operator Learning for Elastic Solids](https://arxiv.org/html/2607.09382)
- Proposes physics-informed operator learning for displacement fields with geometric gaps.
- Inference: High-end research. Our focus on macro-scale flexures is a "down-scaled" but practically applicable version of this.

### 8. [Hybrid ML-Based Beam Solver](https://github.com/sushilmuli/ML-Based-Time-Dependent-Surrogate-FEA-Solver-with-AI-Reporting)
- Predicts stress, deflection, and plastic strain with physics-based time evolution.
- Inference: Shows a trend toward multi-output prediction (stress + deflection), which aligns with our project goals.

### 9. [Accelerating Analysis via DL](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202200099)
- Focuses on the relationship between design parameters and responses.
- Inference: Validates the core "Parametric Geometry -> Response" pipeline.

### 10. [Dynamic Response Surrogate](https://www.researchgate.net/publication/371274609_A_Machine_Learning-Based_Surrogate_Finite_Element_Model_for_Estimating_Dynamic_Response_of_Mechanical_Systems)
- Efficient approach for estimating dynamic response variability.
- Inference: Confirms that modal frequencies (dynamic) and stress (static) can be handled by surrogate models effectively.
