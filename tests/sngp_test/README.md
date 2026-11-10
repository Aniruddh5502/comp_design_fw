---
topic: Noting details about the SNGP text
description: In this file we shall not the eval scripts their functions and how to run them along with all other details of this test.
---

# Introduction
Computing the SNGP architecture for the FEA dataset specially in regression problem. Main target is to see the correlation between predicted error/uncertainity and absolute error.

# MAPPINGS

```text
model/
    └── sngp_best.pth           # Model weights file
scripts/
    |── train.py                 # Trains the SNGP model
    |── evaluate.py              # Evaluates the model
    |── sweep_lipschitz.py       # Compute correlation and other parameters changes with different lipschitz values with a range.
    └── visualize_sweep.py       # Plots the results in plots/ folder
utils/
    |── data_cleaning.py         # Dataset cleaning and dataset making
    └── clear.py                 # Clears the __pycache__/ folders
```

# RUNNING

```bash
cd comp_design_fw/tests
python -m sngp_test.scripts.train               # to train the model
python -m sngp_test.scripts.evaluate            # to evaluate the model
python -m sngp_test.scripts.sweep_lipschitz     # compute the lipschitz data
python -m sngp_test.scripts.visualize_sqweep    # plots the data generated from sqweep
```

# Evaluation Result

```text
└─[>] python -m sngp_test.scripts.evaluate
Model loaded from C:\Users\Administrator\Desktop\code\comp_degp_test\model\sngp_best.pth
Train Data Shape    :   (1327, 10)
Validation Shape    :   (284, 10)
Tests Data Shape    :   (286, 10)
Data loaded successfully.
GP posterior fitted on training set.

=====================================================================
Output     | RMSE       | Avg Std    | Ratio      | Pearson  
---------------------------------------------------------------------
Out 0     | 0.03686    | 0.01026    | 3.593      | 0.193     
Out 1     | 0.03303    | 0.01026    | 3.219      | 0.231     
Out 2     | 0.01646    | 0.01026    | 1.605      | 0.420     
Out 3     | 0.00897    | 0.01026    | 0.874      | 0.462     
Out 4     | 0.01493    | 0.01026    | 1.456      | 0.488     
Out 5     | 0.01501    | 0.01026    | 1.463      | 0.276     
---------------------------------------------------------------------
Mean Spearman Correlation (Abs Error vs Var): 0.2800
=====================================================================
```