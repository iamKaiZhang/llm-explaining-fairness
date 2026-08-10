"""Comparison report -> natural-language explanation.

The explainer sees only the applied modification and the quantitative
comparison report, and is instructed to ground every claim in those
numbers. Faithfulness is enforced by construction: it has no access to
the solver or the data beyond the report.
"""

from __future__ import annotations

import anthropic

from ..scenario import Modification
from . import MODEL

SYSTEM_PROMPT = """\
You explain the outcome of a what-if analysis on a movie recommender system to
the stakeholder who asked the question. The system re-solved its allocation
problem under the requested change and compared the two solutions.

Rules:
- Ground every quantitative claim in the comparison report; never invent
  numbers, movie titles, or effects that are not in it.
- Answer the stakeholder's question first, then give the two or three most
  relevant supporting numbers.
- Explain trade-offs in plain language (e.g. total predicted rating is the
  platform's efficiency objective; exposure is what item providers care about).
- If the report shows no change, say so plainly and explain why.
- Keep it under 200 words. No headers or bullet lists unless the answer has
  genuinely separate parts.
"""


def explain(
    query: str,
    modification: Modification,
    report: str,
    client: anthropic.Anthropic | None = None,
) -> str:
    client = client or anthropic.Anthropic()
    user_content = (
        f"Stakeholder question:\n{query}\n\n"
        f"Change applied to the optimization problem:\n{modification.summary}\n\n"
        f"Comparison report (baseline vs modified solution):\n{report}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined to explain this result")
    return "".join(b.text for b in response.content if b.type == "text")
