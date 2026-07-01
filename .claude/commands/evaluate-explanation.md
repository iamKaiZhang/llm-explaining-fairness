---
description: Score an explanation against the project evaluation dimensions
argument-hint: <path-to-explanation-or-transcript>
---

Evaluate the explanation in: $ARGUMENTS

Score each dimension from 1 to 5 with a one-line justification grounded in the text:

- **Self-consistency**: no internal contradictions across the explanation.
- **Counterfactual robustness**: the "what would change the outcome" claims hold up.
- **Conciseness**: no filler; length matches the decision's complexity.
- **Sufficiency and necessity**: the reasons given actually motivate the decision, and
  each cited reason is load-bearing.
- **Relevance to the user**: pitched to this persona's background and priorities.

Return a short markdown table (dimension, score, justification), then the single
weakest dimension and one concrete fix. Do not reward persuasion; honesty about defects
counts in favor, not against.
