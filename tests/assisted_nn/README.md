---
Title: One NN for output prediction and another one for error mapping
Description: In this architecture we will train a Vanilla NN on a train set and then use a map set to get errors and then train another Vanilla NN to map the difference of Predictor models output and actual output. So this Second model can map the Error Field of the First model and we will test of this error prediction works when we use the test set. In short we shall see if the errors trends are design space driven.
Date: 11/11/2026
Draft: true
---
# Model Predictive Error Manifold Mapping
## 1. Introduction
This project aims to discover a **factual** (non-inferred) indicator of predictive reliability for parametric FEM surrogates to eliminate 'hidden failures'. The core question: *When does an ensemble surrogate get it wrong?*

We hypothesize that errors are **design space driven** — not stochastic in nature. If errors can be predicted structurally, they become a reliable reliability indicator.

**Research Gap**: Previous paths (SNGP, Conformal, Vanilla Ensemble) were falsified:
- SNGP: Variance is a distance-aware inference, not a factual error bound.
- Conformal Prediction: Requires exchangeability; parametric FEM manifolds violate this.
- Vanilla Ensemble: Correlation with error is initialization-dependent (stochastic), not structural.

## 2. Methodology

### Architecture
Two Vanilla NNs with identical architecture:

1. **Predictor NN** (`Primary`): Trained on the train set for output prediction. Input: design space features (4: beam_width, beam_length, fillet_radius, beam_height). Output: 6 target values (max_stress_von_mises, max_deformation, mode_1_freq–mode_4_freq).

2. **Error Mapper NN** (`ErrorMapper`): Trained on the map set to predict the difference between Predictor output and actual output. Input: same feature vector. Output: predicted error field (6 values).

### Error Mapping Process
- **Step 1**: Train Predictor NN on train set → produces predictions (P).
- **Step 2**: For each map set sample, compute actual output (A) and predicted output (P), yielding error field E = A - P.
- **Step 3**: Train ErrorMapper NN on (features, E) pairs → predicts error field from features.
- **Step 4**: On test set, evaluate: compare predicted error (from ErrorMapper) against actual error.

### Mathematical Relationship
Error field E = A - P, where A ∈ ℝ⁶ is actual output, P ∈ ℝ⁶ is Predictor output.
ErrorMapper NN: E = f(θ)(X), where X is the input feature vector, f is the network, θ are learned parameters.

## 3. Experiments

### Data Division
The dataset is split 50:30:10:10 into:
- **Train** (50%): Used to train the Predictor NN
- **Error Map** (30%): Used to generate the error field E = A - P for training ErrorMapper
- **Val** (10%): Used for validation during training
- **Test** (10%): Used for final error prediction evaluation

### Input Data
Source: `data/raw/inputs.csv` — Raw FEM dataset with 10 columns (4 features + 6 targets).
Features: beam_width, beam_length, fillet_radius, beam_height
Targets: max_stress_von_mises, max_deformation, mode_1_freq, mode_2_freq, mode_3_freq, mode_4_freq

### Iterations
The error mapping is tested on the test set. The number of test samples is 10% of the total dataset.

## 4. Results

### Correlation Analysis
- **Pearson correlation** between predicted error and actual error
- **Spearman correlation** (robust to non-linear relationships)
- Reported as a structural fact, not an inference

### Model Performance
- **RMSE**: Root Mean Squared Error for Predictor NN
- **RMSE**: RMSE for ErrorMapper NN (predicting error field)
- **MAE**: Mean Absolute Error for both models

### Quantitative Evidence
- Predictor NN: trained on train set, evaluated on test set
- ErrorMapper NN: trained on error map set, evaluated on test set
- Final metric: correlation between predicted error field and actual error field on test set

## 5. Discussion

The hypothesis that errors are **design space driven** predicts that the ErrorMapper should find structural patterns in the error field — errors that correlate with design parameters rather than being random noise.

**Limitations**:
- The error field E = A - P is a residual, not an independent error measure.
- If the Predictor NN has a systematic bias, the error field will reflect that bias.
- The error mapping is only as good as the Predictor NN's predictions.

**Implications**: A successful error mapper validates that errors are structural (not stochastic), which would prove the reliability indicator is factual rather than inferred.

## 6. Files/Folders

| Path | Description |
|------|-------------|
| `core/models.py` | `Primary` (Predictor NN) and `ErrorMapper` (Error Mapper NN) |
| `utils/data_loader.py` | Data loading, cleaning, scaling, and dataloader creation |
| `data/raw/inputs.csv` | Raw FEM dataset |
| `plots/` | Generated plots from experiments |
| `reports/` | Final analysis reports |

## 7. Running Experiments

### Setup
```bash
# Run data pipeline (loads, scales, creates dataloaders)
cd tests/assisted_nn/utils && python data_loader.py
```

### Train Predictor NN
```bash

```

### Train ErrorMapper NN
```bash

```

### Evaluate
```bash

```

### Full Pipeline (single step)
```bash

```

## 8. Result Mapping

Generated plots are stored in `plots/` directory. There will be some plots once we move once step at a time. The data cleaning pipeline has been made. Now the process progresses as we train the primary model. After that we use the primary network to generate the error map form the error dataset.


<important>
I think we should store the dataloaders once we have made the loaders so that we might train models from it in deterministic way and its reproducable. And also we can keep a choice when we want to randomize the dataloaders so we can see variations in the primary and secondary models.
</important>

---
*Last updated: 11/11/2026 — Draft status. All sections are placeholders pending experimental results.*
