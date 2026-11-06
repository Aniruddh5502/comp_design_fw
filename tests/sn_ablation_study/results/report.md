# Spectral Normalization Capacity Test (multi-seed)

Seeds: [0, 1, 2, 3, 4] | Epochs: 5000 | Adam lr: 0.001 | Width: 64
Metric: final-epoch validation MSE, mean ± std over seeds. Welch t-test against vanilla.

## Target sharpness k = 2.0

Target Lipschitz (grid estimate): 2.101

| Model | Val MSE (mean ± std) | Lipschitz ceiling | Grid Lipschitz | OOD MSE (mean) | p vs vanilla | Verdict |
|---|---|---|---|---|---|---|
| vanilla | 0.00001 ± 0.00001 | none | 2.10 | 0.00834 | — | baseline |
| sn | 0.04274 ± 0.00441 | 1.24 | 1.08 | 0.16316 | 0.0000 | constrained model worse |
| sn_gain5 | 0.01195 ± 0.00121 | 6.28 | 1.87 | 0.09594 | 0.0000 | constrained model worse |

## Target sharpness k = 5.0

Target Lipschitz (grid estimate): 2.870

| Model | Val MSE (mean ± std) | Lipschitz ceiling | Grid Lipschitz | OOD MSE (mean) | p vs vanilla | Verdict |
|---|---|---|---|---|---|---|
| vanilla | 0.00005 ± 0.00005 | none | 2.90 | 0.02448 | — | baseline |
| sn | 0.03731 ± 0.00363 | 1.29 | 1.08 | 0.06193 | 0.0000 | constrained model worse |
| sn_gain5 | 0.02342 ± 0.00129 | 6.25 | 1.56 | 0.08546 | 0.0000 | constrained model worse |

## Target sharpness k = 20.0

Target Lipschitz (grid estimate): 4.823

| Model | Val MSE (mean ± std) | Lipschitz ceiling | Grid Lipschitz | OOD MSE (mean) | p vs vanilla | Verdict |
|---|---|---|---|---|---|---|
| vanilla | 0.00012 ± 0.00018 | none | 4.69 | 0.01848 | — | baseline |
| sn | 0.01875 ± 0.00187 | 1.24 | 0.98 | 0.04858 | 0.0000 | constrained model worse |
| sn_gain5 | 0.01419 ± 0.00236 | 6.16 | 1.04 | 0.00371 | 0.0002 | constrained model worse |
