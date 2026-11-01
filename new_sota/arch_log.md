# Architecture Analysis (Revised)

Scope: surrogate architectures for detecting hidden failures in macro-scale flexure design. Goal: beat SNGP-RFF on hidden-failure detection in this specific case, at a data scale that reaches ~100,000 design points. Prior-art entries are from memory and marked **[verify]** until a literature search is run.

## Definitions

- **Hidden failure:** the surrogate predicts a design as valid (low predicted variance, or a prediction inside the stability region) while the FEA response shows instability or a constraint violation. Failures can lie inside the training input range.
- **Novelty rubric:**
  - **High:** no close prior art found, and a distinguishing experiment exists.
  - **Medium:** a known pattern applied to a new setting, or a new combination with a clear delta.
  - **Low:** functionally equivalent to a known method, or the claimed gain is not testable.
- **Gates (all required):**
  1. **Accuracy:** beats the strongest baseline on the primary metric.
  2. **Scalability:** trains and infers at 100,000 points on a workstation. Exact GP fails this gate.
  3. **Novelty:** at least Medium, with a stated delta from prior art.
- **Baselines for every architecture:** SNGP-RFF, exact or sparse GP (reference at the current ~2,000-run scale only), 20-model MLP ensemble (thesis), and ensemble disagreement as an uncertainty estimate.
- **Primary metric:** hidden-failure recall at a fixed false-alarm rate.
- **Secondary metrics:** negative log-likelihood and interval coverage of the predictive variance; prediction error on ANSYS-verified points.
- **Dataset versions:** record the run count and generation date with every experiment. The thesis count changed from 1,200+ to 1,800–2,000 runs; the 100k set is a planned extension.

Symbols used below:
- $x$: input (beam height $h$, length $L$, fillet radius, width $w$), min-max scaled to $[0,1]^4$
- $\sigma^2_{SNGP}(x)$: SNGP predictive variance
- $p(x)$: normalizing flow density
- $\tilde p(x) = \min\!\left(p(x)/p_{\max},\, 1\right)$: density normalized to training max and clipped to $[0,1]$
- $\alpha \ge 0$: variance-inflation rate, set on held-out data
- $\mathcal{B}$: learned latent region
- $\mathcal{I}$: instability set from FEA
- $g(x) \in [0,1]$: gate value

## Evaluation Plan

- **Two regimes:**
  - *Current set (~2,000 runs):* exact GP included as an accuracy reference.
  - *100k set:* exact GP excluded; comparison is among scalable methods only.
- **Accuracy vs. training-set size:** report primary-metric recall as a function of $n$. An architecture that does not improve with $n$ does not justify the extra data.
- **Scalability column:** training and inference cost as a function of $n$, measured at 100k.

---

## ARCH-01b: Density-Modulated SNGP

**Original claim:** $\sigma^2_{final} = \sigma^2_{SNGP}(x)\cdot \exp(1/p(x))$, rated High.

**Problems:**
1. **Overflow.** $p(x)$ depends on input scaling and is not bounded by 1. A small $p$ gives $\exp(1/p)$ that overflows.
2. **Arbitrary transition.** The inflation onset is set by the units of $p$, not calibrated.
3. **Flow OOD failure.** Flows can assign high likelihood to OOD inputs (Nalisnick et al., 2019 **[verify]**), so gating on $p$ fails where it is most needed.
4. **Prior art.** Density-plus-distance-aware uncertainty is close to DDU (Mukhoti et al., 2023 **[verify]**).

**Revised form:**

$$\sigma^2_{final} = \sigma^2_{SNGP}(x)\cdot\left(1 + \alpha\,(1-\tilde p(x))\right), \quad \alpha \ge 0$$

- Intuition: at the training mode, $\tilde p \approx 1$ and the variance is unchanged. Far from the data, $\tilde p \to 0$ and the variance grows linearly in $\alpha$.
- Clipping $\tilde p \le 1$ keeps the factor $\ge 1$, so the variance cannot go negative for any $\alpha \ge 0$.
- The formula is linear in $\tilde p$, so no exponential overflow arises. Compute $\tilde p$ from $\log p$ when $p$ is very small: $\tilde p = \exp(\min(\log p - \log p_{\max}, 0))$.

**Feasibility:** Low cost. Reuses SNGP plus a normalizing flow on 4D inputs; minibatch training scales to 100k.

**Scalability:** Linear in $n$ (SNGP and flow both train by minibatch SGD).

**Novelty (revised):** Medium-Low. Functional-form change over DDU-style density gating.

**Test:** compare against DDU, SNGP-RFF, ensemble disagreement, and exact GP (current set only) on the primary metric. Report recall at fixed false-alarm rate.

**Status:** Candidate. Priority 1. Needs the DDU comparison before further build work.

---

## ARCH-02b: Manifold-Constrained Latent UQ

**Original claim:** replace fixed output boundary with a learned latent manifold $\mathcal{B}$ of physical instabilities, rated High.

**Problems:**
1. **Labels.** Without instability labels, $\mathcal{B}$ is just the data's latent support, i.e., latent OOD detection (Lee et al., 2018, Mahalanobis **[verify]**).
2. **Overlap with thesis.** The thesis already computes a PCA output manifold (2 PCs ≈ 99% variance). The novelty claim must distinguish learned latent instability from that.
3. **Undefined object.** "Physical instability" is not operationalized.

**Revised form:**

$$\mathcal{B} = \{\, z = f_\phi(x) \;:\; y_{FEA}(x) \in \mathcal{I} \,\}$$

- $f_\phi$: encoder to latent space
- $y_{FEA}$: FEA response
- $\mathcal{I}$: instability set (e.g., pull-in, buckling, stress above yield)
- Requires labeled instability cases from FEA. Without labels, the method reduces to latent OOD.

**Feasibility:** Highest cost. Depends on labeling effort. Encoder training scales to 100k, but label generation may not.

**Scalability:** Linear in $n$ for training; labeling cost is the bottleneck.

**Novelty (revised):** Medium with instability labels; Low without.

**Test:** on a held-out set of hidden-failure cases, compare detection rate against SNGP-RFF, ensemble disagreement, PCA output-manifold baseline, and Mahalanobis latent OOD.

**Status:** Blocked on defining $\mathcal{I}$ and generating labels. Priority 3.

---

## ARCH-03: Manifold-Constrained Ensembles

**Original claim:** specialized training is "essentially importance sampling."

**Problem:** Incorrect. Importance sampling reweights samples to estimate an expectation under a different distribution. Training a specialist on a region changes the model class and training data, not the estimator.

**Revised framing:** region-specialized ensembles. Each specialist is trained on a region $R_k$ of the input space. Specialized ensembles are known in the ensemble literature **[verify]**; the contribution is the region definition.

**Feasibility:** Moderate. Ensemble training is embarrassingly parallel, so the cost is linear in $n$ and in the number of specialists.

**Scalability:** Linear in $n$; inference cost grows with the number of specialists evaluated per input.

**Novelty (revised):** Medium.

**Test:** compare against the thesis 20-model ensemble, SNGP-RFF, and ensemble disagreement on the primary metric.

**Status:** Refined. Rationale corrected; depends on the region definition. Priority 2.

---

## ARCH-03b: Manifold-Gated Ensemble (MGE)

**Original claim:** density-gated routing between generalists and specialists, rated High.

**Problem:** Input-dependent gating is mixture-of-experts (Jacobs et al., 1991; Shazeer et al., 2017 **[verify]**). The density source is the only new element.

**Revised form:**

$$\hat y(x) = g(x)\,\hat y_{spec}(x) + (1-g(x))\,\hat y_{gen}(x), \qquad g(x) = \sigma\!\left(\beta\,(\log \tilde p(x) - \tau)\right)$$

- $\sigma$: logistic function
- $\beta$: sharpness; $\tau$: threshold; both set on validation data
- Intuition: inside high-density regions, trust the specialist; outside, fall back to the generalist.

**Feasibility:** Moderate. Needs the flow, the specialists, and the generalist. The partition and the gate threshold are the main risks.

**Scalability:** Linear in $n$; inference cost is one flow evaluation plus one or two ensemble members per input.

**Novelty (revised):** Medium-Low. Gated MoE with a density gate.

**Test:** compare against a plain MoE with learned gating, the thesis ensemble, and SNGP-RFF on the primary metric.

**Status:** Revised. Priority 2.

---

## Decision Matrix (Revised)

| ID | Novelty | Feasibility | Scalable to 100k | Blocker | Priority |
|---|---|---|---|---|---|
| ARCH-01b | Medium-Low | Low cost | Yes | DDU comparison | 1 |
| ARCH-03 | Medium | Moderate | Yes | Region definition | 2 |
| ARCH-03b | Medium-Low | Moderate | Yes | MoE comparison, gate tuning | 2 |
| ARCH-02b | Medium (labels) / Low | High (labeling) | Training yes, labels bottleneck | $\mathcal{I}$ definition, labels | 3 |

**Next steps:**
1. Define hidden failure and $\mathcal{I}$ with labeled cases.
2. Run prior-art search for DDU, latent OOD, gated MoE, and specialized ensembles. Replace all **[verify]** tags.
3. Implement the ARCH-01b revised form with the DDU and SNGP-RFF baselines on the current set.
4. Run the accuracy-vs-$n$ test and the 100k scalability check on the top candidate.
5. Re-rate after results.