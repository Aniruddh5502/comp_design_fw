---
Title: One NN for output prediction and another one for error mapping
Description: In this architecture we train a Vanilla NN on a train set and then use a map set to get errors and then train another Vanilla NN to map the difference of Predictor model output and actual output. This Second model maps the Error Field of the First model and we test whether this error prediction works when we use the test set. In short we investigate if the error trends are design space driven.
Date: 11/11/2026
Draft: false
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
Two Vanilla NNs with identical architecture (4 → 32 → 32 → 6, ReLU hidden, linear output):

1. **Predictor NN** (`Primary`): Trained on the train set for output prediction. Input: design space features (4: beam_width, beam_length, fillet_radius, beam_height). Output: 6 target values (max_stress_von_mises, max_deformation, mode_1_freq–mode_4_freq).

2. **Error Mapper NN** (`ErrorMapper`): Trained on the map set to predict the difference between Predictor output and actual output. Input: same feature vector. Output: predicted error field (6 values).

### Error Mapping Process
- **Step 1**: Train Predictor NN on train set → produces predictions (P).
- **Step 2**: For each map-set sample, compute actual output (A) and predicted output (P), yielding error field E = A − P.
- **Step 3**: Train ErrorMapper NN on (features, E) pairs → predicts error field from features.
- **Step 4**: On test set, evaluate: compare predicted error (from ErrorMapper) against actual error.

### Mathematical Relationship
Error field E = A − P, where A ∈ ℝ⁶ is actual output, P ∈ ℝ⁶ is Predictor output.
ErrorMapper NN: E = f(θ)(X), where X is the input feature vector, f is the network, θ are learned parameters.

## 3. Experiments

### Data Division
The dataset is split 50:30:10:10 into:
- **Train** (50%): Used to train the Predictor NN
- **Error Map** (30%): Used to generate the error field E = A − P for training ErrorMapper
- **Val** (10%): Used for validation during training
- **Test** (10%): Used for final error prediction evaluation

### Input Data
Source: `data/raw/inputs.csv` — Raw FEM dataset with 12 columns (4 features + 6 targets + 2 metadata).
Features: beam_width, beam_length, fillet_radius, beam_height
Targets: max_stress_von_mises, max_deformation, mode_1_freq, mode_2_freq, mode_3_freq, mode_4_freq
Metadata (dropped by loader): Design_ID, status
Rows: 2000. The loader filters to the 10 required columns and drops rows with missing values before splitting.

### Iterations
The error mapping is evaluated on the test set. The number of test samples is 10% of the total dataset (verified split: 190 test samples).

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
- The error field E = A − P is a residual, not an independent error measure.
- If the Predictor NN has a systematic bias, the error field will reflect that bias.
- The error mapping is only as good as the Predictor NN's predictions.

**Implications**: A successful error mapper validates that errors are structural (not stochastic), which would prove the reliability indicator is factual rather than inferred.

## 6. Files/Folders

| Path | Description |
|------|-------------|
| `core/models.py` | `Primary` (Predictor NN) and `ErrorMapper` (Error Mapper NN) |
| `utils/data_loader.py` | Data loading, cleaning, scaling, and dataloader creation |
| `utils/config.py` | Path/constants definitions for the assisted_nn module |
| `data/raw/inputs.csv` | Raw FEM dataset |
| `data/train_df.csv`, `data/error_df.csv`, `data/val_df.csv`, `data/test_df.csv` | Split datasets (generated by `data_loader.py`) |
| `data/scales.json` | Min-Max scale parameters per column |
| `plots/` | Generated plots from experiments |
| `reports/` | Final analysis reports |

## 7. Running Experiments

### Module Setup
```bash
# Execute any pipeline script from the parent `tests/` directory so relative
# package imports (assisted_nn.*) resolve correctly.
cd tests && python -m assisted_nn.utils.data_loader
```

### Pipeline Steps
| Step | Description | Invoke |
|------|-------------|--------|
| 1. Load & clean | Loads `data/raw/inputs.csv`, drops rows with missing values, filters to the 10 required columns | `load_and_clean_dataset(dataset_raw_file)` |
| 2. Scale | Min-Max scaling per column; writes `data/scales.json` | `scale_normal(loadded_dataframe, data_folder)` |
| 3. Dataloaders | Builds dataloaders from scaled dataframe (or reads stored split CSVs) | `get_dataloaders(scaled_df)` |
| 4. Rebuild from scratch | Forced rebuild via random permutation (dropped the `shuffle=True` flag on the dataloaders so the split itself is what varies), useful for studying variance | `get_dataloaders(scaled_df, build_from_scratch=True)` |

### Reproducibility & Randomization
The `get_dataloaders` function supports both modes:
- `build_from_scratch=False` (default): loads stored CSVs (`data/train_df.csv`, `data/error_df.csv`, `data/val_df.csv`, `data/test_df.csv`) when all four exist — fully reproducible across runs.
- `build_from_scratch=True`: ignores stored CSVs, applies `np.random.seed(seed)` with `seed=41`, then a random permutation to split the dataframe 50:30:10:10. This yields a different split every invocation, allowing study of how split randomness propagates into the Primary and ErrorMapper models.

### Dataloader Parameters
- Batch size: 32
- Shuffle: True
- Device: cpu (override via `device='cuda'` if available)

## 8. Result Mapping

### Status: Data Loading Pipeline Active

The data loading pipeline is implemented and verified. The `data_loader.py` script is ready to:

1. **Load and clean** `data/raw/inputs.csv` — 10 columns (4 features + 6 targets), handle missing values, filter columns.
2. **Scale** data using Min-Max scaling, storing scale parameters in `scales.json`.
3. **Split** the dataset 50:30:10:10 into Train / Error Map / Val / Test.
4. **Create dataloaders** with batch size 32, shuffle enabled.

### Verified Data Splits (raw dataset ~1900 rows)
| Split | Rows (incl. header) | Samples |
|-------|--------------------|---------|
| Train | 949 | 948 |
| Error Map | 570 | 569 |
| Val | 191 | 190 |
| Test | 191 | 190 |

### What's Next
1. **Train Predictor NN** — using the train split (50%) to learn output predictions.
2. **Generate Error Map** — compute error field E = A − P on the error map split (30%), then train the ErrorMapper NN on (features, E) pairs.
3. **Evaluate on Test Set** — compare predicted error against actual error to validate the structural nature of the error field.

### Data Files

| Path | Description |
|------|-------------|
| `core/models.py` | `Primary` (Predictor NN) and `ErrorMapper` (Error Mapper NN) |
| `utils/data_loader.py` | Data loading, cleaning, scaling, and dataloader creation |
| `utils/config.py` | Path/constants definitions |
| `data/raw/inputs.csv` | Raw FEM dataset (10 columns) |
| `data/scales.json` | Min-Max scale parameters per column |
| `plots/` | Generated plots from experiments |

### Current Notes
- Data loading pipeline is implemented and tested.
- Pipeline execution from `tests/` directory correctly resolves relative package imports.
- The `get_dataloaders` function supports both `build_from_scratch=False` (loads stored CSVs) and `build_from_scratch=True` (builds from scratch using random permutation).
- All prior paths (SNGP, Conformal, Vanilla Ensemble) have been falsified. The Error Mapping approach is the current focus.

<important>
I think we should store the dataloaders once we have made the loaders so that we might train models from it in a deterministic way and its reproducible. And also we can keep a choice when we want to randomize the dataloaders so we can see variations in the primary and secondary models.
</important>

---
*Last updated: 11/11/2026 — Data loading pipeline implemented and verified. Experimental training sections (7 & 8) pending next phase.*
