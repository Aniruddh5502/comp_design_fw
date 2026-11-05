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

Now we have tested the constraint of a Di-Lipschitz applied model on the accuracy on same size comparing the vanilla nn and the spectral normalized nn. And found that the vanilla nn is ~66% more accurate in terms of test set validation, just from training.
See [Report](C:\Users\Administrator\Desktop\code\comp_design_fw\new_sota\tests\sn_ablation_study\report_numerical.md)



**Measuring the accuracy cost of spectral normalization in neural surrogates for FEM, and a decoupled architecture that flags unreliable predictions without constraining the predictor**


Now then we can propose a 2 line architecture that has both of the properties. One branch of it will be trained using the vanilla hidden layers and the other one will be trained using the Spectral Normalization applied hidden layers. Now the path which was traiend on using SN, produces thmeasures about how far an input design point is compared to its training data. And the Vanilla 
