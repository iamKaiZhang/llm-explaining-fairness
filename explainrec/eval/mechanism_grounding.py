"""Check that an explanation attributes a change to its true mechanism.

``check_faithfulness`` verifies that an explanation's numbers appear in the
comparison report, but a number can be real while the *cause* the LLM gives
for it is wrong: e.g. attributing a slate change to "this user's predicted
taste changed" when the true mechanism was a constraint being removed, not
any change to the rating model. The report has no field literally labeled
"why", so nothing checks the causal story against it.

This module closes that gap using information the explainer never sees: the
gold ``Modification`` that produced the report tells us, mechanically, which
one of three disjoint mechanisms was exercised (a constraint changed, a
user's rating was re-estimated under a counterfactual attribute, or the
slate size changed). We tag the report with that ground truth ourselves and
then check the explanation's language against a small keyword rubric per
mechanism -- never by asking an LLM, so it stays cheap and auditable, and
never by giving the explainer anything beyond the report it already gets.

This deliberately covers only modifications that touch exactly one
mechanism; a modification that both adds a constraint and overrides a
gender (not used anywhere in ``queries.yaml`` today) has no single true
mechanism to check against and is out of scope (see ``tag_mechanism``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..scenario import Modification

# Vocabulary a grounded explanation of each mechanism should use, and
# vocabulary it should not (that instead belongs to a different mechanism).
# Patterns are matched case-insensitively as substrings; keep them short and
# stem-like ("constraint" catches "constraints", "unconstrained", etc.) so
# minor phrasing changes don't cause false negatives.
_MECHANISM_RUBRIC = {
    "constraint-relaxation": {
        "expected": [
            "constraint", "requirement", "rule", "floor", "cap", "limit",
            "relax", "remov", "loosen", "tighten", "drop",
        ],
        "unexpected": [
            "predicted tastes", "their taste", "rating model", "re-estimat",
            "gender", "demographic", "if they were", "counterfactual",
        ],
    },
    "rating-reestimation": {
        "expected": [
            "gender", "demographic", "attribute", "if they were",
            "counterfactual", "predicted rating for", "taste",
        ],
        "unexpected": [
            "constraint was removed", "constraint was added", "requirement was",
            "rule was", "slate size", "shortened", "lengthened",
        ],
    },
    "slate-size-change": {
        "expected": [
            "slate size", "number of recommendations", "shorten", "lengthen",
            "fewer movies", "more movies", "list size", "k =",
        ],
        "unexpected": [
            "gender", "demographic", "if they were", "constraint was removed",
            "constraint was added",
        ],
    },
}


def tag_mechanism(mod: Modification) -> str | None:
    """Classify a gold ``Modification`` by which single mechanism it exercises.

    Returns ``None`` if the modification is a no-op or touches more than one
    mechanism at once -- there is no single ground truth to check an
    explanation against in either case.
    """
    touches = {
        "constraint-relaxation": bool(mod.add_constraints or mod.remove_constraints),
        "rating-reestimation": bool(mod.gender_overrides),
        "slate-size-change": mod.set_slate_size is not None,
    }
    active = [name for name, hit in touches.items() if hit]
    return active[0] if len(active) == 1 else None


@dataclass
class MechanismResult:
    mechanism: str
    hits: list[str]
    misses: list[str]

    @property
    def grounded(self) -> bool:
        """True if the explanation used at least one expected term and no
        term that belongs to a different mechanism."""
        return bool(self.hits) and not self.misses


_NEGATORS = {"not", "never", "no", "nor"}


def _negated(text: str, start: int) -> bool:
    """True if the match at ``start`` is negated ("was NOT because a
    constraint...") -- a negated mention asserts the opposite, so it should
    count neither for nor against the mechanism."""
    tokens = text[:start].split()[-4:]
    return any(
        t in _NEGATORS or t.endswith("n't")
        for t in (tok.strip(",;:") for tok in tokens)
    ) or " rather than " in " " + " ".join(tokens) + " "


def _find(patterns: list[str], text: str) -> list[str]:
    hits = []
    for p in patterns:
        for m in re.finditer(re.escape(p), text):
            if not _negated(text, m.start()):
                hits.append(p)
                break
    return hits


def check_mechanism_grounding(explanation: str, mechanism: str) -> MechanismResult:
    """Check ``explanation``'s causal language against ``mechanism``'s rubric.

    ``mechanism`` must be one of the keys in ``_MECHANISM_RUBRIC`` (i.e. a
    non-``None`` result of ``tag_mechanism``).
    """
    rubric = _MECHANISM_RUBRIC[mechanism]
    text = explanation.lower()
    hits = _find(rubric["expected"], text)
    misses = _find(rubric["unexpected"], text)
    return MechanismResult(mechanism=mechanism, hits=hits, misses=misses)
