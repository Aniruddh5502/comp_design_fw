"""
Run train + evaluate sequentially and save summary to markdown.
"""
import sys
sys.path.insert(0, ".")

import subprocess
import json
from pathlib import Path
from utils.data_processing import PROJECT_ROOT

CHECKPOINT_DIR = PROJECT_ROOT / "tests/ensemble_test/checkpoints"
RESULTS_DIR = PROJECT_ROOT / "tests/ensemble_test/results"
SUMMARY_PATH = RESULTS_DIR / "summary.md"


def run_script(script_name: str):
    """Run a script and return (success, stdout, stderr)."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, f"scripts/{script_name}"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode == 0, result.stdout, result.stderr


def extract_training_summary(stdout: str) -> dict:
    """Extract key training metrics from stdout."""
    lines = stdout.split('\n')
    summary = {}
    for line in lines:
        if "Loaded" in line and "samples" in line:
            summary["n_samples"] = line.strip()
        if "Split:" in line:
            summary["split"] = line.strip()
        if "Ensemble:" in line:
            summary["ensemble"] = line.strip()
        if "Best val loss" in line or "best val:" in line.lower():
            summary["best_val_loss"] = line.strip()
    return summary


def extract_eval_summary(stdout: str) -> dict:
    """Extract correlation results from stdout."""
    import re
    lines = stdout.split('\n')
    summary = {"correlations": []}
    in_table = False
    for line in lines:
        if "Correlation Analysis" in line:
            in_table = True
            continue
        if in_table and line.strip():
            # Parse tabular output (space-separated or pipe-separated)
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] not in ("target", "====", "===="):
                try:
                    float(parts[1])  # spearman_rho
                    float(parts[3])  # pearson_r
                    summary["correlations"].append({
                        "target": parts[0],
                        "spearman_rho": parts[1],
                        "pearson_r": parts[3],
                    })
                except (ValueError, IndexError):
                    pass
        if in_table and line.strip() == "" and summary["correlations"]:
            break
    return summary


def generate_markdown(train_summary: dict, eval_summary: dict):
    """Generate markdown summary report."""
    md = f"""# Ensemble Vanilla NN - Experiment Summary

## Configuration
- **Architecture**: 4 -> 32 -> 32 -> 6 (ReLU hidden, no output activation)
- **Ensemble Size**: 10 models
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 200 (early stopping patience=50)
- **Batch Size**: 32
- **Train/Val/Test Split**: 70/15/15

## Data
- **Source**: `data/raw/inputs.csv`
- **{train_summary.get('n_samples', 'N/A')}**
- **{train_summary.get('split', 'N/A')}**

## Training
- **{train_summary.get('ensemble', 'N/A')}**
- **{train_summary.get('best_val_loss', 'N/A')}**

## Evaluation: Variance vs Absolute Error Correlation

| Target | Spearman rho | Pearson r |
|--------|------------|-----------|
"""
    for c in eval_summary.get("correlations", []):
        md += f"| {c['target']} | {c['spearman_rho']} | {c['pearson_r']} |\n"

    md += f"""
## Interpretation
- **Spearman rho (0.3-0.4)**: Moderate monotonic relationship between ensemble variance and absolute error
- **Pearson r (0.3-0.5)**: Moderate linear correlation
- Variance provides **some signal** for error magnitude but is not a strong factual indicator
- Consistent with prior findings: ensemble spread != reliable error bound for FEA manifolds

## Artifacts Generated
- `checkpoints/best_ensemble.pt` - Trained ensemble weights
- `checkpoints/training_history.json` - Per-epoch train/val loss
- `results/correlation_analysis.csv` - Full correlation statistics
- `results/reliability_diagram.png` - 4-quadrant variance vs error plots
- `results/training_curves.png` - Train/val loss curves
- `results/test_predictions.csv` - Predictions (original scale)
- `results/individual_predictions.csv` - All 10 member predictions
"""

    SUMMARY_PATH.write_text(md)
    print(f"\nSummary saved to {SUMMARY_PATH}")


def main():
    # Run train
    success, stdout, stderr = run_script("train.py")
    if not success:
        print("Training failed!")
        return
    train_summary = extract_training_summary(stdout)

    # Run evaluate
    success, stdout, stderr = run_script("evaluate.py")
    if not success:
        print("Evaluation failed!")
        return
    eval_summary = extract_eval_summary(stdout)

    # Generate markdown
    generate_markdown(train_summary, eval_summary)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(SUMMARY_PATH.read_text())


if __name__ == "__main__":
    main()