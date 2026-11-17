# Model Predictive Error Manifold Mapping

## Data Loading Pipeline (utils/data_loader.py)
- load_and_clean_dataset dataset_path reads CSV replaces empty strings with NaN drops rows with any missing value keeps required columns
- scale_normal dataframe scale_dir Min Max scaling per numeric column into scales json
- get_dataloaders dataframe seed 41 build_from_scratch False splits data into train error val test 50 30 10 10 using random permutation when build_from_scratch True otherwise loads pre existing CSVs train_df csv error_df csv val_df csv test_df csv Returns dict with keys status content error Inside content are the actual keys train error val test plus train_size error_size val_size test_size
- Run from tests directory cd tests and python -m assisted_nn utils data_loader

## Additional Data Prep Scripts (11/15/2026)
- utils/error_data.py: generate_error_data_loader loads+cleans+scales error_df csv builds error dataloader returns dict with key error. generate_error_mapper_models_training_dataset loads primary.pt runs Primary over error split computes error field E = A - P builds DataFrame [features | targets | error_*] saves to data/error_mapper_training_data.csv returns DataFrame. Prints per-target correlation(actual_target, error_field).
- core/train_error.py: ErrorMap class stub (NEEDS WORK). Instantiates ErrorMapper optimizer MSE loss. get_error_map() method not implemented. Intended to run primary model and send back error map csv data. Not wired into README pipeline.
- Artifacts: data/error_mapper_training_data.csv (11/15), data/error_scales.json (11/15).

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

# Work Progress
## STEP 1: Primary model training
Data preperation is almost direct in this case of primary model. First the dataset is cleaned and made to train, val, error, test dataloaders and saved as CSV. Primary model is trained on the train_df.csv dataset features and targets. And tracked with validation set to ensure its not overfitting on the training data. Plot saved in tests/assisted_nn/plots/ folder.

## STEP 2: Data Preparation
For ErrorMapper() class models data preperation, first doing inference with the error_df.csv using the Primary() model class. Then remarking the target columns with error word to identify that these are prediction errors computed from the error dataset. Now the tests/assisted_nn/utils/error_data.py returns training and validation data unscaled relative to the Primary mdoels data, but scaled normally compared to raw dataset.

## STEP 3: Error Mapper Training
[pending]

## STEP 4: Correlation Study
[pending]


## 11 Last Updated 11/15/2026
- Data loading pipeline implemented and verified
- Primary Trainer implemented and verified end to end
- ErrorMapper trainer implemented via run error mapper py
- Error data prep pipeline (utils/error_data.py) implemented generate_error_mapper_models_training_dataset saves error_mapper_training_data.csv
- README md updated added error_data.py and train_error.py files clarified get_dataloaders output keys removed placeholder usage

# Files
## tests/assisted_nn/core/train.py
Trains the primary model, evaluates it on train and validation data, generates plot saves plot in plots/ folder and prints out the details.
## tests/assisted_nn/core/

