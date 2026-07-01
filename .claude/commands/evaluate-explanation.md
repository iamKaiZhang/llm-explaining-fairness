---
description: Score an explanation against the current evaluation metrics
argument-hint: <path-to-explanation-or-transcript>
---

Evaluate the explanation in: $ARGUMENTS

Read the current metrics from `Experiment Design/Metrics - Explanation Evaluation.md`
fresh (the list is a living draft, so do not rely on a hard-coded set). Score each
adopted or candidate metric from 1 to 5 with a one-line justification grounded in the
text.

Return a short markdown table (metric, score, justification), then the single weakest
metric and one concrete fix. Do not reward persuasion; honesty about defects counts in
favor, not against.
