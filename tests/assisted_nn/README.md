# Model Predictive Error Manifold Mapping

## Data Loading Pipeline (utils/data_loader.py)
- load_and_clean_dataset dataset_path reads CSV replaces empty strings with NaN drops rows with any missing value keeps required columns
- scale_normal dataframe scale_dir Min Max scaling per numeric column into scales json
- get_dataloaders dataframe seed 41 build_from_scratch False splits data into train error val test 50 30 10 10 using random permutation when build_from_scratch True otherwise loads pre existing CSVs train_df csv error_df csv val_df csv test_df csv Returns dict with keys status content error Inside content are the actual keys train error val test plus train_size error_size val_size test_size
- Run from tests directory cd tests and python -m assisted_nn.utils.data_loader

## Additional Data Prep Scripts (11/15/2026)
- utils/error_data.py: get_error_data() loads primary.pt runs Primary over error split computes error field E = |P - A| builds DataFrame [features | targets | error_*] saves to data/error_mapper_training.csv and data/error_mapper_val.csv (same procedure on val_df.csv) returns dict with keys train, val. get_error_dataloaders() builds dataloaders from error data.
- Per-target statistics (MAE, RMSE, nMAE, mean-baseline comparison) are printed by the trainers' evaluate() methods, not by error_data.py.
- Artifacts: data/error_mapper_training.csv (11/15, regenerated 11/19), data/error_mapper_val.csv (11/15, regenerated 11/19). Errors are stored unscaled; features remain Min-Max scaled with scales.json. There is no error_scales.json (config.py references one but nothing writes it).

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
- Step 2 For each map set sample compute actual output A and predicted output P yielding error field E = |P - A|
- Step 3 Train ErrorMapper on (features, E) pairs predicts error field from features
- Step 4 Evaluate on test set compare predicted error (from ErrorMapper) against actual error

### Mathematical Relationship
Error field E = |P - A| where A in R^6 is actual output P in R^6 is Predictor output
ErrorMapper NN: E = f(theta)(X) where X is the input feature vector f is the network theta are learned parameters

## 3 Data Preparation
- Source data raw inputs csv Raw FEM dataset with 12 columns 4 features + 6 targets + 2 metadata
- Features beam_width beam_length fillet_radius beam_height
- Targets max_stress_von_mises max_deformation mode_1_freq mode_2_freq mode_3_freq mode_4_freq
- Metadata (Design_ID status) dropped by loader
- Rows 2000 total after dropping missing values 1897 usable rows

# Work Progress
## STEP 1: Primary model training
Data preperation is almost direct in this case of primary model. First the dataset is cleaned and made to train, val, error, test dataloaders and saved as CSV. Primary model is trained on the train_df.csv dataset features and targets. And tracked with validation set to ensure its not overfitting on the training data. Plot saved in tests/assisted_nn/plots/ folder.

## STEP 2: Data Preparation
For ErrorMapper() class models data preperation, first doing inference with the error_df.csv using the Primary() model class. Then remarking the target columns with error word to identify that these are prediction errors computed from the error dataset. Now the tests/assisted_nn/utils/error_data.py returns training and validation data unscaled relative to the Primary mdoels data, but scaled normally compared to raw dataset.

## STEP 3: Error Mapper Training
Complete. ErrorMapperTrainer class implements full training pipeline with:
- ErrorMapperTrainer class with MSE loss, Adam optimizer
- train() method with epoch tracking, best model saving
- evaluate() method with per-target MAE, RMSE, nMAE statistics
- plot() method for loss curve visualization
- Main execution block runs training on error_mapper_training.csv and then calls evaluate() on the val loader
- Models and plots were regenerated on 11/19/2026 (note: trainer prints "[PRIMARY TRAINER]" tags and uses non-_error target names in its __main__ evaluate() call - cosmetic only)

## STEP 4: Correlation Study
Partial. Per-target error statistics computed on train and validation sets. Model beats baseline on all targets.
The actual test-set correlation study (Spearman and Pearson correlations between the true error E = |P - A| and the ErrorMapper-predicted error, per target column) is NOT implemented yet. src/main.py documents the intended final stage and src/test.py contains an unfinished scaffold for it.


## STEP 5: Test-set Evaluation (pending)
To be implemented in src/test.py and orchestrated by src/main.py:
1. create_dataloader_from_test() builds a deterministic (shuffle=False) dataloader from data/test_df.csv - implemented (not yet verified end to end)
2. get_actual_error_df() - load Primary, predict on the test set, compute E_actual = |P - A| - scaffold, incomplete (model loading line is broken, preds overwritten per batch, returns nothing)
3. get_predicted_error_df() - pass test features through ErrorMapper to get E_pred - stub (pass)
4. Compute Spearman and Pearson correlations per target column between E_actual and E_pred and report them

## 11 Last Updated 11/19/2026
- Data loading pipeline implemented and verified
- Primary Trainer implemented and verified end to end
- ErrorMapper trainer implemented and verified via python -m assisted_nn.core.trainer_errormapper
- Error data prep pipeline (utils/error_data.py) implemented
- get_error_data() and get_error_dataloaders() functions implemented
- ErrorMapper training completed with per-target statistics
- src/main.py rewritten (11/18): old orchestrator removed, now just the spec for the final test-set evaluation stage
- src/test.py added (11/19): create_dataloader_from_test() implemented; get_actual_error_df()/get_predicted_error_df() unfinished
- Trainers' __main__ blocks now call evaluate() after training (11/19)
- Models, error-mapper CSVs and plots regenerated (11/19)
- README updated to reflect actual implementation in codebase (run_error_mapper.py and error_scales.json references removed)

# Files
## tests/assisted_nn/src/main.py
Spec (docstring only, 11/18/2026) for the final stage: run Primary on the test set, compute actual error E = |P - A|, run ErrorMapper on the same test features, then compute Spearman and Pearson correlations between actual and predicted error per target column. The old orchestrator code was removed; implementation is pending in src/test.py.

## tests/assisted_nn/src/test.py
Final test-set evaluation scaffold (added 11/19/2026). create_dataloader_from_test() builds a non-shuffled test dataloader from data/test_df.csv - implemented (has a __main__ smoke-test block, not yet run). get_actual_error_df() is incomplete: it uses torch.load_state_dict (does not exist; should be model.load_state_dict), the batch loop overwrites preds instead of accumulating, and the function falls through to pass. get_predicted_error_df() is a stub. Spearman/Pearson correlation computation is not implemented yet.

## tests/assisted_nn/core/models.py
Defines Primary and ErrorMapper neural network classes. Both use identical architecture: 4 -> 32 -> 32 -> 6 with ReLU activations.

## tests/assisted_nn/core/train.py
PrimaryTrainer class. Trains primary model on train split, validates on val split. Implements run_epoch(), plot(), train(), evaluate() methods. Saves primary.pt model weights. The __main__ block trains and then runs evaluate() on the val loader (added 11/19/2026).

## tests/assisted_nn/core/trainer_errormapper.py
ErrorMapperTrainer class. Trains error mapper on error_mapper_training.csv, validates on error_mapper_val.csv. Implements run_epoch(), plot(), train(), evaluate() methods. Saves error_mapper.pt model weights. The __main__ block trains and then runs evaluate() on the val loader (added 11/19/2026). Note: __main__ prints "[PRIMARY TRAINER]" tags and evaluates with non-_error target names - cosmetic only.

## tests/assisted_nn/utils/config.py
Path configuration constants (data_folder, model_folder, plots_folder, dataset_raw_file, etc.) and color constants (accent, focus, success, another).

## tests/assisted_nn/utils/data_loader.py
Data loading pipeline: load_and_clean_dataset(), scale_normal(), get_dataloaders(). Handles CSV loading, Min-Max scaling, train/val/error/test splits (50:30:10:10).

## tests/assisted_nn/utils/error_data.py
Error data preparation: get_error_data(), get_error_dataloaders(). Runs Primary model on error_df, computes error field E = |P - A|, saves error_mapper_training.csv and error_mapper_val.csv.

## tests/assisted_nn/data/
Data artifacts: train_df.csv, error_df.csv, val_df.csv, test_df.csv, error_mapper_training.csv, error_mapper_val.csv, scales.json.

## tests/assisted_nn/model/
Trained model weights: primary.pt, error_mapper.pt.

