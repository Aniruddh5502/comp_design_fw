# Model Predictive Error Manifold Mapping

## 1. Introduction
This project investigates whether errors in a parametric FEM surrogate are **design space driven** (structural) rather than stochastic. We hypothesize that if errors are predictable via a second Neural Network, they become a factual reliability indicator for the surrogate.

**Research gap**: Previous paths (SNGP, Conformal, Vanilla Ensemble) were falsified:
- SNGP: Variance is a distance-aware inference, not a factual error bound.
- Conformal Prediction: Requires exchangeability; parametric FEM manifolds violate this.
- Vanilla Ensemble: Correlation with error is initialization-dependent (stochastic), not structural.

## 2. Methodology

### Architecture
Two Vanilla NNs with identical architecture (4 -> 32 -> 32 -> 6, ReLU hidden, linear output):

1. **Predictor NN** (`Primary`): Trained on the train set for output prediction. Input: design space features (beam_width, beam_length, fillet_radius, beam_height). Output: 6 target values (max_stress_von_mises, max_deformation, mode_1_freq->mode_4_freq).
2. **Error Mapper NN** (`ErrorMapper`): Trained on the error map split to predict the difference between Predictor output and actual output. Input: same feature vector. Output: predicted error field (6 values).

### Error Mapping Process
- **Step 1**: Train `Primary` on train set -> produces predictions (P).
- **Step 2**: For each map-set sample, compute actual output (A) and predicted output (P), yielding error field E = A - P.
- **Step 3**: Train `ErrorMapper` on (features, E) pairs -> predicts error field from features.
- **Step 4**: Evaluate on test set: compare predicted error (from ErrorMapper) against actual error.

### Mathematical Relationship
Error field E = A - P, where A in R^6 is actual output, P in R^6 is Predictor output.
ErrorMapper NN: E = f(theta)(X), where X is the input feature vector, f is the network, theta are learned parameters.

## 3. Data Preparation
- Source: `data/raw/inputs.csv` -- Raw FEM dataset with 12 columns (4 features + 6 targets + 2 metadata).
- Features: beam_width, beam_length, fillet_radius, beam_height
- Targets: max_stress_von_mises, max_deformation, mode_1_freq, mode_2_freq, mode_3_freq, mode_4_freq
- Metadata (Design_ID, status) dropped by loader.
- Rows: 2000 total -> after dropping missing values -> 1897 usable rows.

### Data Loading Pipeline (utils/data_loader.py)
- `load_and_clean_dataset(dataset_path)`: reads CSV, replaces empty strings with NaN, drops rows with any missing value, keeps required columns.
- `scale_normal(dataframe, scale_dir)`: Min-Max scaling per numeric column into `scales.json` (stores min/max per column).
- `get_dataloaders(df, seed=41, ...)`: splits data into train/error/val/test (50:30:10:10) using random permutation when build_from_scratch=True; otherwise loads pre-existing CSVs (train_df.csv, error_df.csv, val_df.csv, test_df.csv). Returns dataloaders dict.
- Feature columns: beam_width, beam_length, fillet_radius, beam_height. Target columns: max_stress_von_mises, max_deformation, mode_1_freq, mode_2_freq, mode_3_freq, mode_4_freq.

### Verified Verified Splits (2000 rows -> 1897 after dropna)
- Train: 949 rows (948 samples)
- Error Map: 570 rows (569 samples)
- Val: 191 rows (190 samples)
- Test: 191 rows (190 samples)

## 4. Running Experiments

### Module Setup
```bash
cd tests && python -m assisted_nn.utils.data_loader
```

### Pipeline Steps
- Step 1: Load & clean -> `load_and_clean_dataset(dataset_raw_file)`
- Step 2: Scale -> `scale_normal(loadded_dataframe, data_folder)`
- Step 3: Dataloaders -> `get_dataloaders(scaled_df)`
- Step 4: Rebuild from scratch -> `get_dataloaders(scaled_df, build_from_scratch=True)`

### Reproducibility and Randomization
- Default (build_from_scratch=False): loads stored CSVs -> fully reproducible across runs.
- build_from_scratch=True: forces fresh random permutation (seed=41 internal) -> different splits each invocation, enabling study of how split randomness propagates into models.

### Dataloader Parameters
- Batch size: 32
- Shuffle: True
- Device: CPU (use device='cuda' if CUDA available).

## 5. Primary Trainer (core/train.py)

### Implementation
- File: core/train.py
- Class: PrimaryTrainer -- trains `Primary` Predictor NN on train split, validates on val split.
- 100 epochs, Adam optimizer (lr=1e-3), MSE loss, device="cpu", seed=41.
- Saves best model (lowest val loss) to model/primary.pt; reloads best weights at end.
- Prints per-epoch train/val loss (epoch 1, every 10 epochs, final).
- Generates plots/Primary-Training-Loss.svg.

### Verified Behaviour
- End-to-end run succeeds.
- Best Val Loss ~0.0002 MSE (Min-Max scaled [0,1] range); Final Train Loss ~0.0001; Val ~0.0002.
- Per-epoch logs show convergence.

### Code Review Findings
- optimizer.zero_grad() inside train=True branch of run_epoch -- correct.
- torch.enable_grad() correctly gates gradients -- correct.
- Primary and ErrorMapper classes defined in core/models.py -- correct.

## 6. ErrorMapper Trainer (core/trainer_errormapper.py -- PENDING)

### Implementation (NOT YET WRITTEN)
- Follow PrimaryTrainer skeleton, freezing Primary's weights, computing error field E = A - P on error-map split, train ErrorMapper on (features, E) pairs, evaluate on test set, produce correlations/RMSE/MAE.

### Usage
```bash
cd tests && python -m assisted_nn.core.train  # placeholder; needs ErrorMapperTrainer
```

## 7. ErrorMapper Class (core/models.py)
Already defined:
```python
class ErrorMapper(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=None, output_dim=6):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [32, 32]
        layers = []
        previous_dimension = input_dim
        for dimension in hidden_dim:
            layers.append(nn.Linear(previous_dimension, dimension))
            layers.append(nn.ReLU())
            previous_dimension = dimension
        layers.append(nn.Linear(previous_dimension, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```
Needs a trainer.

## 8. Result Mapping

### Status: Data Loading Pipeline Active
Verified: data_loader.py executes from tests/ dir, resolving imports correctly.

### Verified Data Splits (raw dataset 2000 rows, 1897 post-dropna)
- Train: 949 rows (948 samples)
- Error Map: 570 rows (569 samples)
- Val: 191 rows (190 samples)
- Test: 191 rows (190 samples)

## 9. Progress

- Primary Trainer (core/train.py): Implemented and verified end-to-end. Saves model/primary.pt, plots training loss.
- ErrorMapper Trainer: Not yet written. Needs to compute error field on error-map split and train ErrorMapper.
- Next: implement ErrorMapperTrainer, evaluate on test set (correlation/Pearson/Spearman; RMSE/MAE).

## 10. Files/Folders

- core/models.py: Primary (Predictor NN) and ErrorMapper (Error Mapper NN)
- utils/data_loader.py: Data loading, cleaning, scaling, dataloader creation
- utils/config.py: Path/constants (dataset_raw_file, data_folder, colors)
- data/raw/inputs.csv: Raw FEM dataset (10 columns)
- data/train_df.csv, data/error_df.csv, data/val_df.csv, data/test_df.csv: split datasets
- data/scales.json: Min-Max scale params per column
- plots/: Plotting outputs (e.g., Primary-Training-Loss.svg)
- reports/: Final analysis reports (future)

## 11. Last Updated: 11/11/2026
- Data loading pipeline implemented and verified.
- Primary Trainer implemented and verified end-to-end.
- ErrorMapper trainer pending next phase.
- README.md previously contained stale/high-level description; now aligned with code.
