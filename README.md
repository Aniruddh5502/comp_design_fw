# When Does an Ensemble Surrogate Get It Wrong?

> Research plan for the flexure surrogate-optimization work, in plain language. One question, three claims, a few checks, and a step-by-step plan.

---

## 1. The Question

**When the optimizer picks a design, can we still tell from the ensemble's disagreement that the surrogate might be wrong? And does this get better with more training data?**

Here is the setup:

1. We train a surrogate on sampled designs.
2. The optimizer (NSGA-II) searches the surrogate for the best designs.
3. Those best designs are _chosen_ because the surrogate likes them, often near constraint limits. They are not random.

On random test data, high disagreement usually goes with high error. But the optimizer's picks are special. The link between disagreement and error may break exactly there.

**How to frame the paper:** show what the data says and when the surrogate fails. Do not present a new method. If the link turns out to hold, that is still a useful result.

---

## 2. Why This Direction

Some of the old plan is already well studied, so it would not count as new.

|Topic|What already exists|What it means for us|
|:--|:--|:--|
|Disagreement vs. error|Disagreement between ensemble members has been used to estimate error. Later work says ensemble calibration can get worse as disagreement grows ([Kirsch & Gal](https://arxiv.org/abs/2202.01851)).|Mostly classification. Don't claim "disagreement tracks error" as new.|
|Ensembles under data shift|One study found ensemble uncertainty mostly comes from single-model uncertainty, so ensembles add little extra ([NeurIPS 2022 review](https://gresearch.com/news/neurips-2022-paper-review-2)).|We need a simple baseline to show disagreement adds value (Check A).|
|Ensemble vs. GPR|Deep ensembles tend to be underconfident, and post-hoc calibration fixes this. They were compared with GPR ([Yang & Lee](https://arxiv.org/abs/2303.16210)).|GPR vs. ensemble is not new. Use GPR only as a comparison.|
|Ensembles in optimization|Uncertainty from ensembles has been used inside optimization search ([Audet et al.](https://arxiv.org/abs/2107.04360)).|Don't propose a new "uncertainty-aware optimizer."|
|Optimizers exploiting surrogate errors|Surrogates often over-promise outside their training data and mislead the optimizer ([Tan et al.](https://arxiv.org/abs/2410.11502)).|Supports our idea. But it is mostly ML benchmarks, not structural design.|
|"Variance goes up with error"|Reported routinely, for example in a recent fusion surrogate ([arXiv 2608.24785](https://arxiv.org/pdf/2608.24785)).|A baseline, not a finding.|

> **Note:** I found no paper doing exactly this for flexure surrogates, but I only ran a few searches. That is not proof. Do a full literature search before you claim a gap.

**Gap in one sentence (draft):** _Disagreement and error are usually compared on random test data, but design optimization uses chosen designs. We test whether the link still holds there, and how often the surrogate fails without warning, compared with ANSYS._

---

## 3. The Three Claims

Each claim can be proven wrong by the data.

### Claim 1: The link breaks at the optimizer's picks

For the same level of disagreement, the surrogate's error is larger at optimizer-chosen designs than at random test designs.

- **How to test:** combine the ANSYS-checked optimizer designs and a random test subset. Fit `log(error) ~ log(disagreement) + chosen + log(disagreement):chosen`.
- **Supported if:** the `chosen` term (or the interaction) is positive and its bootstrap 95% interval does not include zero.
- **Wrong if:** the interval includes zero. Then disagreement works fine, and that is worth reporting.

### Claim 2: Hidden failures change with training-set size

A **hidden failure** is when the error is above a tolerance τ but the disagreement is low (in the lowest quarter of test-set disagreement). The rate of hidden failures changes in a clear way as training size N grows.

- **For each N, report:** hidden-failure rate, how well disagreement detects errors above τ (AUROC), Spearman correlation, and the N where the error-vs-N power law stops holding.
- **Wrong if:** the hidden-failure rate stays near zero at every N.

### Claim 3: Some "feasible" designs break constraints

Some designs the surrogate calls feasible break the constraints in ANSYS. Adding a safety margin of `k × disagreement` removes most of these, at a measurable cost to the objective.

- **For each N, report:** the share of predicted-feasible designs that fail in ANSYS, the failures left after the margin, and how much the objective gets worse.
- **Wrong if:** almost no designs fail even without the margin.

---

## 4. Checks Reviewers Will Ask For

|Check|Why|
|:--|:--|
|**A. Distance-to-training-data baseline** (kNN distance in the 4-input space)|If disagreement is no better than "how far is this from the training data," the ensemble adds nothing. This is the most important check.|
|**B. One GPR baseline**|A comparison only, not a claim.|
|**C. 3 to 5 training seeds per N**|Separates the effect of N from random training noise.|
|**D. Nested training subsets** (N1 inside N2 inside ...) and one fixed test set|Differences come from N, not from which samples were picked. Set aside the test set first.|
|**E. Fix thresholds in advance** (τ, the quartile cut, k)|Prevents tuning them after seeing results.|
|**F. Optional: one analytic benchmark with a known answer**|Shows the result is not only about one device. Do it only if cheap.|

---

## 5. Rules for Running the Experiments

1. **Choosing the ~100 ANSYS designs.** Spread them across the Pareto front and near constraint limits. Optimizer outputs cluster together, so you effectively have fewer than 100 independent points. Use a block bootstrap for the intervals.
2. **ANSYS is not reality.** It has its own mesh error. Check it on a few designs with a finer mesh, so it is not confused with surrogate error. Say this limit clearly in the paper.
3. **Define error once.** Use the same error definition (relative or absolute) for every output.
4. **Look at failed simulations.** Plot where they fall in the design space. If they cluster at extreme shapes, the dataset is biased where it matters.
5. **Report every N and every seed.** No picking only the good checkpoints.

---

## 6. Step-by-Step Plan

|Step|What to do|Move on when|
|:--|:--|:--|
|**0. Literature check**|Full search. Write the one-sentence gap.|The gap still looks real. If not, rethink the scope.|
|**1. Setup**|Fix the test set. Build nested subsets for 5 values of N. Fix τ, the quartile cut, and k.|The thresholds are written down before any ANSYS checks.|
|**2. Train**|Train a 20-member ensemble for each N, with 3 to 5 seeds. Save checkpoints. Add Checks A and B.|Test-set error and disagreement are computed.|
|**3. Test-set analysis** (no new ANSYS cost)|Disagreement vs. error for each N. Find where the power law breaks. Compare with Check A.|The basic results look sensible.|
|**4. Pilot**|Run the optimizer on one checkpoint. Check about 20 chosen designs near constraint limits in ANSYS.|You can see which way Claim 1 points. If it clearly does not hold, rethink before spending more runs.|
|**5. Full check**|Run the optimizer on each checkpoint. Check about 100 designs per N. If runs are limited: 100 at two values of N and about 40 at the others.|You have enough points for reliable intervals.|
|**6. Write the paper**|Evidence first. Label your own interpretations as interpretations.|Every claim links to a figure and an interval.|

---

## 7. Dropped or Postponed

|Item|Why|
|:--|:--|
|Physics-based features (Euler-Bernoulli) as a calibration claim|Feature engineering is not new. It would need a baseline with the same features. Maybe later.|
|Thick-beam "honesty" test|Hard to separate from plain data shift. Revisit after the main claims.|
|Uncertainty-aware optimizer and regret analysis|A separate paper. Existing work overlaps.|
|PICP × MPIW called an "Interval Score"|Not the standard definition. If you report interval quality, use the Winkler interval score, plus CRPS or NLL, and coverage at several levels.|
|Terms like "Trust-Gap" and "Death-Match"|Use standard words: calibration, coverage, hidden failure, out-of-distribution detection.|

---

## 8. Where to Publish and What Could Go Wrong

- **Journals that fit:** applied engineering and design-optimization journals, such as _Structural and Multidisciplinary Optimization_ or _Engineering with Computers_. Check their current scope and ranking first.
- **Risk 1:** with only four inputs and a smooth response, disagreement may just work well everywhere. Claim 1 covers this. If the link breaks, that is the interesting result. If it holds, you have confirmed standard practice.
- **Risk 2:** only one device. Check F, or an honest statement of where the results apply, helps.
- **Writing:** a clear claim backed by strong evidence matters more than a new method.

---

## 9. Quick Summary

- **One question:** does disagreement warn us about errors at optimizer-chosen designs, across training sizes?
- **Three claims:** the link breaks at chosen designs; hidden failures change with N; some "feasible" designs break constraints.
- **Key checks:** kNN-distance baseline, several seeds, nested subsets, thresholds fixed in advance.
- **Start with:** literature check, then test-set analysis, then a 20-design ANSYS pilot, then the full check.
- **Don't claim:** that variance tracks error, that ensemble vs. GPR is new, or that you built a new framework.