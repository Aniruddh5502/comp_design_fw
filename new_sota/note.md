---
title: Searching the research topic
description: This is a note for me to keep doing the research and find the research topic in computational mechancical Engineering.
date: 30/10/2026
---

# Computational Mechanical Engineering(CME)
Its not a combination of topics, moreover its an *Approach*. Using Numeraical methods to model, simulate, and solve mechanical egnieering problems that are too complex for hand calculations.

## The Core Pillers
Accross almost all programms these foundational areas can be found.
- Numerical Methods and Mathematics
- Computational Solid Mechanics(CSM): The structure side. The FEA is dominant tool here. 
- Computational Fluid Dynamics(CFD): The flow side. This focuses on the Finite Volume Method.
- Multiphysics & Coupling: Real-World problems often involve interacting physics.

## Modern Extensions & Specializations
- AI & Machine Learning: Using NN networks for consitutive laws, building surrogae models to accelerate FEM simulations, and physics informed neural nets(PINNs).
- Advanced Mechanics: Courses on Feacture Mechanics, damage mechanics etc
- *Optimization*: Gradient Based optimization and inverse problems, often coupled with FEM or AI to design better systems.
- Application Domains: biomechanics (implants, tissue), acoustics, crash simulation, or energy systems.


# SNGP Study

- Paper [Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness](https://arxiv.org/abs/2006.10108)

## Review
This paper applies the SNGP architecture to classification models. Not regression models or continious output models. So the fundamental thing doesn't match.

And there are variations of this SNGP on continious output models. Used on some other fields. But none of them has been applied on the MEMS flexure or normal flexure or accelerometer class geometry. To predict FEA results and solid mechanics Physics.

So this is a new take, even if we don't modify the archtecture

what if we keep 2 paths. One with Spectral Normalization that keeps the distance awareness mappings right from the inputs to the hidden layers and it ends with predicting the distance with the mean predictions.

And the other one with normal NN, only predicts the mean. And we calculate MSE error on both of them. 

Now if both models goes to achieve same accuracy with large enough/const training compute budget/iteration. Then what does the scaling case does different to approximate the true function.

And if both of them doesn't reach the same MSE error performance then who does better and what does the spectral Normalization changes in the network beyond the ||h-h_train|| and ||x-x_train||.
