---
title: Testing a combination of model types to make a Ensemble of different kinds of model architectures.
description: This paper makes a new ensemble with all kinds of top performing models and uses its behaviour to make the better ensemble.
link: https://www.emergentmind.com/topics/ensemble-surrogate-model
---
## Problem: 
Expensive or complex simulations (such as PDEs, fluid dynamics, and environmental models) are computationally prohibitive to run repeatedly. Single-model predictive surrogates often struggle with accuracy, lack robust uncertainty quantification (UQ), and may fail to generalize across diverse data distributions.

## Hypothesis: 
By strategically aggregating multiple diverse predictive surrogates (e.g., Gaussian processes, neural networks, decision trees) into an ensemble, one can improve overall predictive accuracy, provide a reliable measure of epistemic uncertainty (via model disagreement), and enhance the robustness of the system compared to any single surrogate model.

## Experiment: 
Researchers implemented various ensemble strategies across multiple scientific domains:
 • Construction: Using weighted aggregations (bagging, stacking), packed-ensemble architectures (grouped convolutions), and conditional normalizing flows (SurroFlow).
 • Uncertainty Quantification: Applying variance/spread-based measures, deep evidential regression with conformal prediction (ConfEviSurrogate), and ensembles of conditional GANs.
 • Application Areas: Testing these models on ocean modeling (FNOs), fluid flow around airfoils, sea-ice concentration forecasting, and adversarial attack transferability.

## Results:
 • Higher Accuracy: Ensemble models (e.g., in ocean modeling) yielded
   lower RMSEs than individual surrogates.
 • Robust UQ: Ensemble variance provided a usable estimate of epistemic
   uncertainty, which improved exploration in Bayesian optimization.
 • Efficiency: Packed-ensemble architectures and vectorized processing mitigated the computational overhead of running multiple models.
 • Improved Transferability: In adversarial settings increasing ensemble diversity and size demonstrably reduced transfer error and increased attack success rates.

## Review: 
Ensemble surrogate models are a powerful framework for
scientific computing because they balance the trade-off between
computational cost and predictive reliability. The most critical factor
for success is diversity; if member models are too similar, the
uncertainty estimates become overconfident and unreliable. While
computational overhead remains a challenge, modern architectural
optimizations (like parameter sharing) make these models viable for
high-performance computing (HPC) environments.