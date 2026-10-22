# Red-Team Log: NN Ensemble UQ

## Self-Audit Questions
1. **Did I search for refutation as hard as for support?**
   - Yes. I specifically searched for "limitations of SNGP" and "Deep Ensembles OOD failure". I found that Deep Ensembles are often NOT effectively robust compared to single models in some benchmarks.
2. **Do my sources truly come from independent origins?**
   - Yes. Sources include TensorFlow official tutorials, Nature, NeurIPS, arXiv, and Springer.
3. **Is my best evidence recent and from matching conditions?**
   - Yes. Most sources are 2021-2026. The "4-input" condition is addressed by the fact that low-dimensional inputs make distance-based methods (SNGP, Mahalanobis) computationally efficient and more likely to be effective.
4. **Am I mistaking a popular opinion for evidence?**
   - I almost did with "Calibration". I initially looked for TS/Platt, but the research showed they are for classification. I corrected this by finding the Nature paper on bootstrap regression calibration.
5. **Could survivorship bias hide failures?**
   - Possibly. Most papers report successes. However, the "Deep Ensembles Work, But Are They Necessary?" paper provides a critical counter-point to the efficacy of NNEs for OOD.
6. **Did I drift away from the original question?**
   - No. I stayed focused on the "Proxy UQ" and "Deception clusters" problem.
7. **Does every claim in my draft match its evidence in strength?**
   - Yes. SNGP is labeled as "High" confidence for distance-awareness because it is the core design goal of the architecture. Calibration is "Low" for OOD because it relies on the model's own (potentially flawed) uncertainty signal.

## Identified Gaps
- I have not found a direct "Head-to-Head" benchmark between SNGP and NN-GP Residuals for a structural design problem. I am inferring the comparison based on the properties of the methods.
- I have not found "Trust-Region" UQ constraints in the way the user suggested; they are optimization tools. I must report this as a "Misconception" or "Limit".
