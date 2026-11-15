# Model Predictive Error Manifold Mapping

## Data Loading Pipeline (utils/data_loader.py)
- load_and_clean_dataset dataset_path reads CSV replaces empty strings with NaN drops rows with any missing value keeps required columns
- scale_normal dataframe scale_dir Min Max scaling per numeric column into scales json
- get_dataloaders dataframe seed 41 build_from_scratch False splits data into train error val test 50 30 10 10 using random permutation when build_from_scratch True otherwise loads pre existing CSVs train_df csv error_df csv val_df csv test_df csv Returns dict with keys status content error Inside content are the actual keys train error val test plus train_size error_size val_size test_size
- Run from tests directory cd tests and python -m assisted_nn utils data_loader

## Code Review Findings
- Optimizer gradient gating etc confirmed correct

## 1 Introduction
This project investigates whether errors in a parametric FEM surrogate are design space driven (structural) rather than stochastic We hypothesize that if errors are predictable via a second Neural Network they become a factual reliability indicator for the surrogate

Research gap Previous paths SNGP Conformal Vanilla Ensemble were falsified
- SNGP Variance is a distance aware inference not a factual error bound
- Conformal Prediction requires exchangeability parametric FEM manifolds violate this
- Vanilla Ensemble Correlation with error is initialization dependent (stochastic) not structural

## 2 Methodology

### Architecture
Two Vanilla NNs with identical architecture 4 -> 32 -> 32 -> 6 ReLU hidden linear output

1 Primary NN Primary Trained on the train set for output prediction Input design space features beam_width beam_length fillet_radius beam_height Output 6 target values max_stress_von_mises max_deformation mode_1_freq mode_2_freq mode_3_freq mode_4_freq

2 Error Mapper NN ErrorMapper Trained on the error map split to predict the difference between Predictor output and actual output Input same feature vector Output predicted error field 6 values

### Error Mapping Process
- Step 1 Train Primary on train set produces predictions P
- Step 2 For each map set sample compute actual output A and predicted output P yielding error field E = A - P
- Step 3 Train ErrorMapper on (features E) pairs predicts error field from features
- Step 4 Evaluate on test set compare predicted error (from ErrorMapper) against actual error

### Mathematical Relationship
Error field E = A - P where A in R^6 is actual output P in R^6 is Predictor output
ErrorMapper NN: E = f(theta)(X) where X is the input feature vector f is the network theta are learned parameters

## 3 Data Preparation
- Source data raw inputs csv Raw FEM dataset with 12 columns 4 features + 6 targets + 2 metadata
- Features beam_width beam_length fillet_radius beam_height
- Targets max_stress_von_mises max_deformation mode_1_freq mode_2_freq mode_3_freq mode_4_freq
- Metadata (Design_ID status) dropped by loader
- Rows 2000 total after dropping missing values 1897 usable rows

### Data Loading Pipeline (utils/data_loader.py)
- load_and_clean_dataset dataset_path reads CSV replaces empty strings with NaN drops rows with any missing value keeps required columns
- scale_normal dataframe scale_dir Min Max scaling per numeric column into scales json
- get_dataloaders splits data into train error val test (50 30 10 10) using random permutation when build_from_scratch True otherwise loads pre existing CSVs Returns dict with keys train error val test

## 4 Running Experiments

### Module Setup
cd tests and python -m assisted_nn utils data_loader

### Pipeline Steps
- Step 1 Load clean load_and_clean_dataset raw_input csv
- Step 2 Scale scale_normal df data_folder
- Step 3 Dataloaders get_dataloaders scaled_df
- Step 4 Rebuild from scratch get_dataloaders scaled_df build_from_scratch True

### Reproducibility and Randomization
- Default (build_from_scratch False) loads stored CSVs fully reproducible across runs
- build_from_scratch True forces fresh random permutation (seed 41 internal) different splits each invocation

### Dataloader Parameters
- Batch size 32
- Shuffle True
- Device CPU (use device cuda if CUDA available)

### Data Splits (after cleaning & scaling)
- Train 949 rows (948 samples)
- Error Map 570 rows (569 samples)
- Val 191 rows (190 samples)
- Test 191 rows (190 samples)

## 5 Primary Trainer (core/train.py)

### Implementation
- File core/train.py
- Class PrimaryTrainer trains Primary Predictor NN on train split validates on val split
- 100 epochs Adam optimizer (lr 1e-3) MSE loss device cpu seed 41
- Saves best model (lowest val loss) to model primary pt reloads best weights at end
- Prints per epoch train val loss (epoch 1 every 10 epochs final)
- Generates plots Primary Training Loss svg

### Verified Behaviour
- End to end run succeeds
- Best Val Loss ~0.0002 MSE Final Train Loss ~0.0001 Val ~0.0002
- Per epoch logs shown

### Code Review Findings
- optimizer zero grad inside train True branch of run_epoch correct
- torch enable grad correctly gates gradients correct
- Primary and ErrorMapper classes defined in core models py correct

## 6 ErrorMapper Trainer (core/trainer_errormapper.py)

### Implementation
- Follows PrimaryTrainer skeleton freeze Primary weights compute error field E = A - P on the error map split train the ErrorMapper on (features E) pairs evaluate on test set produce correlations RMSE MAE

### Usage
cd tests and python assisted_nn core run error mapper py
(This script orchestrates load scale dataloaders initialize ErrorMapperTrainer train plot)

### Model Class (core/models.py)
- ErrorMapper vanilla NN default hidden 32 32 ReLU linear output to 6 targets

## 7 ErrorMapper Class (core/models.py)
Already defined

## 8 Result Mapping

### Status Data Loading Pipeline Active Verified
- utils data_loader py runs from tests dir resolves imports correctly
- get_dataloaders returns dict with keys train error val test

### Verified Data Splits (raw dataset 2000 rows 1897 after dropna)
- Train 949 rows (948 samples)
- Error Map 570 rows (569 samples)
- Val 191 rows (190 samples)
- Test 191 rows (190 samples)

## 9 Progress

- Primary Trainer (core/train.py) Implemented and verified end to end Saved model primary pt plotted loss
- ErrorMapper Trainer Implemented via run error mapper py Will train ErrorMapper on error map split evaluate on test set
- Next Run ErrorMapperTrainer compute correlation Pearson Spearman RMSE MAE on test set validate structural error trend mapping onto design space

## 10 Files Folders

- core models py Primary (predictor NN) and ErrorMapper (error mapper NN) classes
- utils data_loader py Data loading cleaning scaling dataloader creation
- utils config py Path constants (dataset_raw_file data_folder plots_folder colors)
- data raw inputs csv Raw FEM dataset
- data train_df csv error_df csv val_df csv test_df csv
- data scales json Min Max scale params per column
- model primary pt error mapper pt (outputs)
- plots Primary Training Loss svg ErrorMapper Training Loss svg (plots)
- run error mapper py End to end runner for ErrorMapperTrainer

## 11 Last Updated 11/14/2026
- Data loading pipeline implemented and verified
- Primary Trainer implemented and verified end to end
- ErrorMapper trainer implemented via run error mapper py
- README md updated added run error mapper py command clarified get_dataloaders output keys removed placeholder usage