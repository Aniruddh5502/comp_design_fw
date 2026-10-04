---
Title: Literature Review
Description: Guide to do Proper Literature Review
---

## STEP BY STEP

1. Start from what your data can answer, not from the literature. Your dataset is the asset. Write two or three concrete questions it could settle, such as how ensemble error behaves near the constraint limits, or whether surrogate error on optimizer-selected designs is larger than on random test points. A gap is a question you can answer with evidence that others haven't shown.[instance](instance/dataset_eval.md)

2. Find 3 to 5 seed papers, then chase citations. Find the closest papers in Google Scholar or Semantic Scholar. Go backward through their references and forward through "cited by". Connected Papers or ResearchRabbit help you see the cluster. This finds more than keyword searching, because different fields name the same thing differently ("surrogate-assisted optimization", "metamodel", "infill", "optimizer exploitation of surrogate error").

3. Read in this order: abstract, figures and tables, conclusion and limitations, then methods only if it matters. Most of the time you only need to know what they did, with how much data, validated how.

4. Keep a comparison matrix with one row per paper: problem, dataset size, method, validation approach, metrics, stated limitations. *Gaps show up as empty cells, such as "nobody validated on optimizer outputs", or as papers that disagree with each other.*

5. Mine limitations and future-work sections. Authors often state the gap outright.

6. Stress-test a gap before trusting it. Search it five or six ways, check the last two or three years of forward citations, and ask why nobody did it. Sometimes it's because it's hard or uninteresting.
