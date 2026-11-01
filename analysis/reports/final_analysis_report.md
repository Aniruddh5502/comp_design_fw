
# FINAL DATA ANALYSIS REPORT
======================================================================

## 1. Dataset Summary
- **Source:** data/processed/full_processed_dataset.csv
- **Total Samples:** 1820
- **Total Features:** 12

## 2. Profiling Results

--- CSV Profile: full_processed_dataset.csv ---
Total Rows: 1820
Total Columns: 12
Empty Rows: 0
------------------------------
Col #   Header               Type       Nulls    Min          Max         
---------------------------------------------------------------------
0       Design_ID            int64      0        1            2000        
1       beam_width           float64    0        0.2013688722879548 1.999947632811813
2       beam_length          float64    0        10.0017117591943 39.97573020615898
3       fillet_radius        float64    0        0.1009545147605954 1.999229677628755
4       beam_height          float64    0        5.000267675912441 19.9931295840198
5       status               str        0        N/A          N/A         
6       max_stress_von_mises float64    0        0.0939508571288682 13.842807670326136
7       max_deformation      float64    0        0.0021283109832839 4.947364481720527
8       mode_1_freq          float64    0        23.02788820149861 1139.2662734493913
9       mode_2_freq          float64    0        31.202172347275955 1399.9096245875346
10      mode_3_freq          float64    0        49.60955596653007 2018.3103491664792
11      mode_4_freq          float64    0        51.09410939366549 2248.947758690331


## 3. Visual Analysis Findings
- **Correlations:** A detailed correlation matrix has been generated. Strong relationships between geometry (width, height) and target metrics (stress, deformation) are evident.
- **Distributions:** Individual histograms have been plotted for all numerical features. Mean and Median lines are included to identify skewness.

## 4. Artifacts Location
- **Profiling Report:** analysis/reports/profiling_report.txt
- **Visualizations:** analysis/plots/
- **Analysis Scripts:** analysis/scripts/
======================================================================
