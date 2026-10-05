
[**Vardhan et al. 2024, Ocean Engineering 311, 118777**](https://www.sciencedirect.com/science/article/pii/S0029801824021152)
UUV hull drag: BO vs other optimizers, and a DNN surrogate

- Problem: minimize drag of a Myring hull; each RANS CFD run is expensive
- Dataset size: 3021 valid CFD runs (3333 generated); 2160 train / 240 val / 621 test
- Method: OpenFOAM (k-ω SST, 2 m/s). Six optimizers compared with CFD in the loop; MLP surrogate (22k params) with GA in the loop
- Validation approach: random held-out test split; surrogate-in-the-loop vs CFD-in-the-loop on 4 cases; no experimental data
- Metrics: sample efficiency, run-to-run variance; MAPE 1.85%, 96.9% within 5%, 2.25% outliers (>10%)
- Stated limitations: needs many sims to train; small UUVs, single flow condition; outliers where data is sparse; no architecture ablation
