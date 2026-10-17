# OOD Candidate Analysis Report (Deterministic)

## 1. Objective Performance
- **Min Target Error ($|f_1 - 100|$):** 0.00359 Hz
- **Mean Target Error ($|f_1 - 100|$):** 59.575 Hz
- **Max $f_2/f_1$ Ratio:** 12,068,161.91
- **Mean $f_2/f_1$ Ratio:** 415,823.87

## 2. Physical Validity Audit
- **Predicted $f_1$ Mean:** 95.108 Hz
- **Predicted $f_1$ Min:** -3.969e-07 Hz
- **Predicted $f_2$ Mean:** 162.450 Hz
- **Predicted $f_2$ Min:** 7.279 Hz
- **Negative Frequency Counts:**
    - $f_1$ (out_2): 2 candidates
    - $f_4$ (out_4): 18 candidates
    - $f_5$ (out_5): 22 candidates

## 3. Design Space Boundary Analysis
- **Beam Width Max:** 1.99999 (Limit: 2.0)
- **Beam Length Max:** 39.99999 (Limit: 40.0)

## 4. Conclusion
The surrogate produces mathematically optimal but physically impossible results. The extreme $f_2/f_1$ ratio and negative frequencies in the top-performing candidates prove that the surrogate's internal logic collapses in the OOD boundary regions.
