
# Test Report: Spectral Normalization (SN) Ablation Study

## 1. Objective
Evaluate the impact of Spectral Normalization on the predictive accuracy (MSE) of a neural surrogate model compared to a vanilla neural network. This test determines the "Reliability Tax"—the loss in accuracy incurred by enforcing a Lipschitz constraint for distance-awareness.

## 2. Experimental Setup
- **Dataset**: Synthetic 2D response surface simulating FEA behavior (Smooth global trend + Sharp local peak).
- **Training Set**: 1500 samples | **Validation Set**: 500 samples.
- **Architectures**:
    - **VanillaNet**: 3-layer MLP with SiLU activations.
    - **SNNet**: 3-layer MLP with Spectral Normalization applied to all Linear layers and SiLU activations.
- **Constants**:
    - Learning Rate: $0.001$ (Adam)
    - Epochs: 500
    - Network Width: 64 units per hidden layer.
    - Seed: 42 (Deterministic).

## 3. Results

| Model | Final Val MSE | Performance Gap |
| :--- | :--- | :--- |
| VanillaNet | 0.0148 | Baseline (1.0x) |
| SNNet | 0.0246 | ~1.6660x Error With SN |

## 4. Analysis
- **Finding**: The Vanilla network significantly outperformed the SN-constrained network.
- **Reasoning**: The Spectral Normalization constraint $\sigma(W) \le 1$ prevents the network from utilizing the weight magnitudes necessary to encode high-curvature regions (sharp peaks) of the response surface.
- **Conclusion**: In the context of FEA surrogates, strict Spectral Normalization acts as a bottleneck that hinders peak accuracy.

## 5. Implications for Architecture
The "Reliability Tax" is measured at approximately 66% increase in MSE. To maintain high accuracy while achieving reliability, the architecture must pivot from a single constrained network to a decoupled approach:
1. **Mean Predictor**: Unconstrained (Vanilla) for maximum accuracy.
2. **Reliability Certifier**: SN-constrained for distance-aware uncertainty signaling.

## 6. Evidence Path
- **Script**:   C:\Users\Administrator\Desktop\code\comp_design_fw\new_sota\tests\sn_ablation_study\probe_sn.py
- **Log**   :   C:\Users\Administrator\Desktop\code\comp_design_fw\new_sota\tests\sn_ablation_study\results 
- **Plot**  :   C:\Users\Administrator\Desktop\code\comp_design_fw\new_sota\tests\sn_ablation_study\results\ablation_plots.svg 

