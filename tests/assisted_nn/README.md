# README for assisted_nn

## Overview
This repository contains a Neural Network implementation for parametric Finite Element Method (FEM) surrogate modeling. It implements a two-stage neural network architecture: a Primary NN for output prediction and an Error Mapper NN for predicting the error field between the predicted output and the actual output.

## Architecture
The system consists of a Primary NN and an Error Mapper NN.
- **Primary NN**: Takes design space features as input and predicts the 6 target values (e.g., `max_stress_von_mises`, `max_deformation`, etc.).
- **Error Mapper NN**: Takes the same design space features as input and predicts the error field $E = |P - A|$ between the predicted output $P$ and the actual output $A$.

## Methodology
1. **Data Preparation**: Raw FEM data is loaded, cleaned, scaled, and split into training, validation, test sets.
2. **Primary NN Training**: The Primary NN is trained on the training set to predict the target outputs.
3. **Error Mapping**: The Error Mapper NN is trained on the error map derived from the Primary NN's predictions.
4. **Evaluation**: The final stage involves calculating correlation (Pearson and Spearman) between the actual error field $E_{actual}$ and the predicted error field $E_{predicted}$ for each target column.

## Key Findings
- The Primary NN demonstrates strong performance across all targets.
- The Error Mapper NN successfully predicts the error field with high correlation (e.g., Pearson correlation $\approx 0.89$ for `max_stress_von_mises`).
- This suggests that errors in the surrogate model are primarily structural rather than stochastic.

## Data Flow
Raw Data $\rightarrow$ Data Loader $\rightarrow$ Primary NN $\rightarrow$ Predicted Output ($P$) $\rightarrow$ Error Calculation ($E = |P - A|$) $\rightarrow$ Error Mapper NN $\rightarrow$ Predicted Error Field ($E_{pred}$)

## Dependencies
- Python 3.14.7
- Libraries (e.g., NumPy, TensorFlow/PyTorch)

## Usage
To run the full pipeline, execute `python -m assisted_nn.src.main`.
