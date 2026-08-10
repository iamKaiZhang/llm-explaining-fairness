"""Natural-language query -> structured ``Modification``.

Uses structured outputs (``client.messages.parse`` with the Pydantic
``Modification`` schema), so the LLM cannot emit an edit the optimizer
does not understand; every accepted modification is re-validated by the
constraint builders before solving.
"""

from __future__ import annotations

import anthropic

from ..scenario import Modification, Scenario
from . import MODEL

SYSTEM_PROMPT = """\
You translate stakeholder questions about a movie recommender system into a
structured modification of its underlying optimization problem.

The system solves:
  maximize   sum over users u and movies i of r_hat[u,i] * x[u,i]
  subject to sum_i x[u,i] = slate_size for every user (each user gets a slate),
             0 <= x[u,i] <= 1,
             plus the named constraints listed below.
r_hat[u,i] is the predicted 1-5 star rating. The rating model contains an
explicit gender pathway, so gender counterfactuals are supported via
gender_overrides.

Dataset: {dataset_summary}

Current slate size: {slate_size}
Currently active constraints:
{constraints}

Rules:
- Express the question as the smallest edit that answers it. Do not add
  constraints the question does not ask about.
- "Stop exploring" / "no cold-start promotion" means removing the active
  cold-item exposure constraint (and only if the question implies it, also
  forbidding cold items).
- Questions of the form "would user u be recommended differently if they were
  male/female" map to a gender_overrides entry and that user in focal_users.
- Constraint names must be new, short, and kebab-case.
- User and item indices are 0-based.
- If the question requires no change to the problem (it asks about the current
  solution), return an empty modification and say so in the summary.
"""


def interpret(
    query: str,
    scenario: Scenario,
    client: anthropic.Anthropic | None = None,
) -> Modification:
    client = client or anthropic.Anthropic()
    system = SYSTEM_PROMPT.format(
        dataset_summary=scenario.data.summary(),
        slate_size=scenario.slate_size,
        constraints=scenario.describe_constraints(),
    )
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": query}],
        output_format=Modification,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined to interpret this query")
    if response.parsed_output is None:
        raise RuntimeError("could not parse a Modification from the model output")
    return response.parsed_output
