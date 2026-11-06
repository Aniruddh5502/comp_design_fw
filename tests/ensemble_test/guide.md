# Ensemble Test Guide

## Purpose
Tests whether **ensemble variance** (spread across 10 independently trained neural networks) correlates with **absolute prediction error** on parametric FEA surrogates. This evaluates if ensemble spread is a factual reliability indicator for engineering surrogate models.

## Folder Structure
```
tests/ensemble_test/
├── core/
│   ├── model.py          # Ensemble class (10x MLP), forward pass, predict_with_uncertainty
│   └── trainer.py        # Training loop, early stopping, checkpointing, dataloaders
├── scripts/
│   ├── train.py          # Train ensemble from scratch (200 epochs, patience=50)
│   ├── evaluate.py       # Evaluate on test set; generate plots + CSVs
│   └── run_all.py        # Run train.py -> evaluate.py sequentially
├── utils/
│   └── data_processing.py  # Load/clean/split CSV; scaling; inverse transform
├── checkpoints/
│   ├── best_ensemble.pt      # Trained weights (10 models)
│   └── training_history.json # Per-epoch train/val loss
├── results/
│   ├── correlation_analysis.csv   # Spearman/Pearson per target
│   ├── reliability_diagram_*.png  # 6 plots: variance vs abs error (4 quadrants)
│   ├── training_curves.png        # Log-scale loss curves
│   ├── correlation_heatmap.png    # Heatmap of correlations across 6 targets
│   ├── error_distribution.png     # Error histograms per target
│   ├── test_predictions.csv       # Mean pred, variance, true, abs error (original scale)
│   ├── individual_predictions.csv # All 10 ensemble members per sample
│   └── summary.md                 # Auto-generated experiment summary
└── data/
    └── (train/val/test splits created at runtime from data/raw/inputs.csv)
```

## What the Code Tests
| Hypothesis | Test |
|------------|------|
| Ensemble variance correlates with absolute error | Spearman & Pearson correlation per target |
| Variance identifies "confident but wrong" predictions (Q3: low var, high error) | Reliability diagram quadrant analysis |
| Ensemble generalizes on parametric FEA manifold | Test MAE/RMSE on held-out 15% |

**Targets (6):**
- `max_stress_von_mises`
- `max_deformation`
- `mode_1_freq` — `mode_4_freq`

**Architecture:** 10 independent MLPs (4 → 32 → 32 → 6), ReLU hidden, no output activation.  
**Optimizer:** Adam (lr=1e-3).  
**Data:** 1897 samples from `data/raw/inputs.csv` → 70/15/15 split.

## Where is the Report
- **Auto-generated summary:** `results/summary.md` (created by `run_all.py`)
- **Correlation statistics:** `results/correlation_analysis.csv`
- **Visual reports:** 6 reliability diagrams + training curves + heatmap + error distributions
- **Raw predictions:** `results/test_predictions.csv`, `results/individual_predictions.csv`

## Latest Results (from most recent run)

### Correlation: Variance vs Absolute Error

| Target | Spearman ρ | Pearson r |
|--------|-----------|-----------|
| max_stress_von_mises | 0.226 | 0.596 |
| max_deformation | 0.648 | 0.396 |
| mode_1_freq | 0.476 | 0.508 |
| mode_2_freq | 0.353 | 0.318 |
| mode_3_freq | 0.405 | 0.389 |
| mode_4_freq | 0.341 | 0.382 |

### Interpretation
- **Spearman ρ (0.23–0.65)**: Moderate monotonic relationship — variance somewhat tracks error rank
- **Pearson r (0.32–0.60)**: Moderate linear correlation — stronger for stress/deformation
- **Key finding**: Ensemble variance provides *some* signal but is **not a strong factual error bound** for FEA manifolds
- **Danger zone (Q3)**: Low variance + high error cases exist — ensemble can be confidently wrong

### Training
- Best validation loss: **0.000076** (MSE, normalized scale)
- 10 models × 14,140 params each
- Early stopping at epoch ~180 (patience=50)

*Full details in `results/summary.md`*

## How to Run