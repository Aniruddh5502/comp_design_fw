---
title: Experiment Descisions
description: Holds results and summary for each test cases that are runned.
---
# Tests

## 1. sn_ablation_study
### Hypothesis
Does Spectral Normalization effect the Learning ability of NNs if enought compute is given, inferring some information is to be stored in the weights scaling.
### Result
True. SN does badly effect the NNs learning ability. Compared to Vanilla NNs leaning easily and achieving very low error floor in test data.
### Findings
1. Spectral Normalization restricts the model from learning the representations and restricts its expressiveness for FEA/Physics simulations data.

## 2. SNGP Test In Distrivution FEA dataset
### Hypothesis
Weather SNGP can predict variance for FEA data and relate this variance with abs error with high correlation.
### Result
False. The Correlation came to about ~0.2 both spearman and pearson. So they aren't correlated.
### Findings
1. SNGP's prediction variance doesn't align with error calculated over the test set. The spearman correlation and pearson correlation is very low ~0.2, so their correlation is not enough strong to conclude that one can be predicted from the other reliably/confidence.

2. SNGP was made for classification data to detect OOD cases(On which classes the model wasn't trained on.) It doesn't work in FEA data where continious data is the norm.

