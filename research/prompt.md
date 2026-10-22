---
title: prompts
description: This is a prompt collection for research sub tasks such as paper sumarization(yet I could come up with this task, I have used llms for other reasons, but couldn't do it reliably, maybe it was some prompting mistake that I didn't specify the terms clearly.)
---

# Paper Sumarization

fetch this link and sumarize this paper into these sections
title: [title](url)
<format>

## problem:
From the paper, at what problem does the paper comes into conclusion and vauches to provide a solve for.

## Baseline: 
If they have used the SOTA setup in current date (current time is {time}). write description of which methods have they used.

## Experiment
what experiment have they used in which specification.
dataset: scale, columns, varity, specification of the dataset they have used that was used for training, validation etc.

## Results
what results they have found. If that satisfies the claim they have made. and if that's justified. weather there are any logical flaw.

## Review
what conclusion have they drawn from the results, is the conclusion justified.

## Gaps
If they have stated any future work(run searches again if someone has covered this gap that this paper has satted), gap that has not been covered in that study.

</format>

<additional_instructions>
- if the link is a arxiv pdf file, use the pdf parser tool.
- if the link is html, use the fetch tool to fetch the content.
</additional_instructions>

<>