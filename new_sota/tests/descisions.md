# Architectural Decisions

## SN Ablation Study: The Reliability Tax
**Source**: `new_sota/tests/sn_ablation_study/`

### Premise
Quantify the accuracy loss caused by Spectral Normalization (SN) in parametric FEA surrogates. The hypothesis is that the Lipschitz constraint required for distance-awareness limits the network's ability to fit high-curvature physical features.


### Results
- **Vanilla MLP MSE**: $\approx 0.0122$
- **SN-MLP MSE**: $\approx 0.0203$
- **Relative Error Increase**: $\approx 44\%$

### Conclusion
Strict SN imposes a "Reliability Tax." It prevents weight scaling that is essential for encoding complex physical gradients. Distance-awareness and peak predictive accuracy are mutually exclusive in a single-stream network.

### Decision
Decouple function approximation from reliability certification.
- **Mean Path**: Use a Vanilla MLP for maximum accuracy.
- **Reliability Path**: Use SNGP to provide a distance-aware variance $\sigma^2$.
- **Integration**: Calibrate $\sigma^2$ to actual error $\delta$ via an external mapping function.


## Hybrid Arch Study
### Requirements(Must hold before results mean anything)
1. Each model is evaluated on the same data splits, with the same seeds, in box and out-of-box.
2. Vanilla and SNGP are each trained to convergence or a fixed budget. For fair comparison.
3. The hybrid's predictor is identical to vanilla's-only the flag branch is added. Otherwise the hybrid accuracy cannot be attributed to the design.
4. The flag is scored against vanillas actual error, not against its own mean.
5. All numbers in the reports should be computed in code, not typed in.

### Results
1. SNGP doesn't work on FEA data, it might be suitable for classification data but for FEA data its performance and error/variance spearman correlation is inconclusive
2. Its just a glorified embedding model stacked onto a NN.
