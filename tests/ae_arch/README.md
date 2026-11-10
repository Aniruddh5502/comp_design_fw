---
title: Auto Encoder architecture test
description: In this repository we shall test the variance, abs error spearman and pearson correlation for Autoencoder Architecture for ensembles. And then we can comapre the results of this correlations with Vanilla NN Ensemble Networks correlation
date: 08/11/2026
---

# IDEA

1. First Do PCA Analysis on the input data variability. Determine min DOF for 100%.
2. Train a Encoder Decoder Net with min DOF bottleneck, loss is reconstruction loss of input params.
3. 

<thinking>

# The target
The target is to somehow deterministically capture the error in and show a correlation between prediction of variance and actual error upper than 0.5 or something in FEA simulation surrogates.

# plan
- train a encoder-decoder network to reconstruct the input data from min 100% capturing PCA DOF. Now this encoder net can represent the latent space that's trained on the input space. Rather than like SNGP that needs to constrint the weights to keep the input mapping intact which in terms degrades the performance in terms of actual accuracy. And also SNGP is for classification data and OOD identification. But here we shall be working with in range In Distribution or Out Of Distribution(not from the same peobabilistic distribution of the training data).

Well and conformal mapping also cannot work when there is less correlation between the prediction and abs error.
Now I need to think, how do I can make such predictions that are aware of the error fields of the model. Or maybe take the DOF that the input data have like from the Encoders latent output dimension. And map how models each output changes from changing those DOFs.
</thinking>

# TESTS
- Did PCA test turned out each variabce has enough proportions on its axis to contribute enough information to the input variance.
- To test yourself run `python -m ae_arch.pca_test.pca_test` from `comp_design_fw/tests`

# MAP
config/                 # Configuration files, paths, settings, state
    config.py           # Manages the settings.json, state.json and any other file
    config.json         # JSON file holding the settings.
utils/                  # Utility functions for data processing, modeling, training
    data.py             # Loads, Cleans, Scales, Makes Train, Val, Test dataloaders
data/                   # Holds csv files, data scaling info
    scales.json         # Scales per column for dataset scaling and rescaling.
ae_tests/               # contains the tests for the code written for this project
models/                 # contains the encoder-decoder model and the predictor network.
    en_net.py           # encoder decoder network class
    vanilla_nn.py       # vanilla nn predictor class(the predictor class)
    
# RESULT
- IDEA DROPPED.