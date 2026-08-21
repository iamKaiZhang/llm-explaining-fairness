"""Comparison report -> natural-language explanation.

The explainer sees only the applied modification and the quantitative
comparison report, and is instructed to ground every claim in those
numbers. Faithfulness is enforced by construction: it has no access to
the solver or the data beyond the report.
"""

from __future__ import annotations

from ..scenario import Modification
from .backend import ApiBackend, Backend

SYSTEM_PROMPT = """\
You explain the outcome of a what-if analysis on a movie recommender system to
the stakeholder who asked the question. The system re-solved its allocation
problem under the requested change and compared the two solutions.

Rules:
- Ground every quantitative claim in the comparison report; never invent
  numbers, movie titles, or effects that are not in it.
- When the report includes a focal user's profile (age, gender, occupation,
  rating history), use it to personalize the answer. The profile shows
  recorded attributes; if the applied change describes a counterfactual
  override, distinguish the two rather than treating them as a contradiction.
- If the question is asked by or about a specific user (a focal profile is in
  the report), write the answer for that person and match the language to
  their age and occupation. For quantitative occupations (engineer,
  programmer, scientist) technical vocabulary like "constraint",
  "optimization objective", or "Gini" is fine; for most other occupations
  prefer everyday phrasing ("rules the system must follow", "movies with
  little rating history", "how unevenly this is spread"), giving the key
  numbers without jargon; for teenagers use short, concrete sentences. Adapt
  register and depth only: the facts and numbers stay exactly the same, and
  never assume interests or ability beyond word choice.
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
    backend: Backend | None = None,
    include_change_summary: bool = True,
) -> str:
    """Explain the comparison report to the stakeholder.

    ``include_change_summary=False`` withholds the applied change from the
    prompt, so the explainer must infer the cause from the report alone.
    This is an evaluation ablation (does the mechanism-grounding score
    survive when the cause is not handed to the model?), not a production
    mode -- the demos and ``Pipeline.ask`` always include the summary.
    """
    backend = backend or ApiBackend()
    change_block = (
        f"Change applied to the optimization problem:\n{modification.summary}\n\n"
        if include_change_summary
        else (
            "Change applied to the optimization problem: (withheld for this "
            "run; infer what changed only from the comparison report)\n\n"
        )
    )
    user_content = (
        f"Stakeholder question:\n{query}\n\n"
        f"{change_block}"
        f"Comparison report (baseline vs modified solution):\n{report}"
    )
    return backend.text(SYSTEM_PROMPT, user_content)
