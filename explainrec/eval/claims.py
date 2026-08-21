"""Decompose an explanation into atomic directional claims and verify each.

``check_faithfulness`` verifies that the *numbers* in an explanation appear
in the comparison report, but not that they are attached to the right story:
"cold item exposure rose to 3775" passes the number check even when the
report shows exposure *falling* to 0 (3775 is the base value). This module
closes that gap by extracting atomic claims of the form

    <metric> went <direction>        e.g. "total predicted rating fell"

and verifying each against the report's actual base -> modified movement.
The result is a graded per-explanation score (fraction of directional claims
the report supports) instead of one binary verdict, so "right cause, wrong
story" is distinguishable from "wrong everywhere".

Like the other checks in this package this is a pure function over a
(report, text) pair -- no LLM judge -- so it is fast, unit-testable, and
cheap enough for the live demo. The price is recall: only metrics in the
lexicon and directions stated with common movement verbs are extracted;
paraphrases outside the lexicon produce no claim rather than a wrong one.
Claims about metrics the report does not carry (e.g. a distribution that a
solve did not produce) are marked unverifiable and excluded from the score
rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# metric lexicon: surface phrase -> metric label -> (base, modified) paths
# ---------------------------------------------------------------------------

# Report subtree each metric's numbers must live under, for the field
# attribution check: a number stated next to a metric phrase must match a
# value under that metric's own subtree, not merely appear somewhere in the
# report.
_SUBTREE_PATHS: dict[str, tuple[str, ...]] = {
    "objective": ("objective",),
    "mean_predicted_rating": ("mean_predicted_rating",),
    "cold_item_exposure": ("cold_item_exposure",),
    "slate_size": ("slate_size",),
    "exposure_gini": ("item_exposure_concentration", "gini"),
    "top_share": ("item_exposure_concentration", "top_10pct_items_exposure_share"),
    "worst_off": ("per_user_slate_rating_distribution",),
    "exploration_gini": ("exploration_burden_distribution",),
}

# (base path, modified path) into the compare_solutions() report dict.
_PAIR_PATHS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "objective": (("objective", "base"), ("objective", "modified")),
    "mean_predicted_rating": (
        ("mean_predicted_rating", "base"), ("mean_predicted_rating", "modified"),
    ),
    "cold_item_exposure": (
        ("cold_item_exposure", "base_total"), ("cold_item_exposure", "modified_total"),
    ),
    "slate_size": (("slate_size", "base"), ("slate_size", "modified")),
    "exposure_gini": (
        ("item_exposure_concentration", "gini", "base"),
        ("item_exposure_concentration", "gini", "modified"),
    ),
    "top_share": (
        ("item_exposure_concentration", "top_10pct_items_exposure_share", "base"),
        ("item_exposure_concentration", "top_10pct_items_exposure_share", "modified"),
    ),
    "worst_off": (
        ("per_user_slate_rating_distribution", "base", "min"),
        ("per_user_slate_rating_distribution", "modified", "min"),
    ),
    "exploration_gini": (
        ("exploration_burden_distribution", "base", "gini"),
        ("exploration_burden_distribution", "modified", "gini"),
    ),
}

# Longest phrases first so "cold item exposure" wins over "cold item".
_METRIC_PHRASES: list[tuple[str, str]] = sorted(
    [
        ("total predicted rating", "objective"),
        ("objective", "objective"),
        ("efficiency", "objective"),
        ("mean predicted rating", "mean_predicted_rating"),
        ("average predicted rating", "mean_predicted_rating"),
        ("average rating", "mean_predicted_rating"),
        ("cold item exposure", "cold_item_exposure"),
        ("cold-item exposure", "cold_item_exposure"),
        ("exposure of cold items", "cold_item_exposure"),
        ("cold items", "cold_item_exposure"),
        ("cold item", "cold_item_exposure"),
        ("exploration burden", "exploration_gini"),
        ("exposure concentration", "exposure_gini"),
        ("concentration", "exposure_gini"),
        ("gini", "exposure_gini"),
        ("top 10%", "top_share"),
        ("top 10 percent", "top_share"),
        ("worst-off", "worst_off"),
        ("worst off", "worst_off"),
        ("slate size", "slate_size"),
        ("number of recommendations", "slate_size"),
    ],
    key=lambda p: -len(p[0]),
)

# Generic phrases suppressed when they trail a more specific metric mention
# ("the exploration burden gini rose" is one claim about exploration_gini,
# not a second claim about exposure_gini).
_GENERIC_PHRASES = {"gini", "concentration"}
_GENERIC_SUPPRESS_GAP = 12  # chars

_DIRECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("up", re.compile(
        r"\b(increas(?:e|es|ed|ing)|rose|rises?|rising|grew|grows?|growing"
        r"|higher|climbed|doubled|went up|goes up)\b")),
    ("down", re.compile(
        r"\b(decreas(?:e|es|ed|ing)|fell|falls?|fallen|dropped|drops?"
        r"|declin(?:e|es|ed|ing)|lower|shrank|reduced|halved|went down"
        r"|goes down)\b")),
    ("flat", re.compile(
        r"\b(unchanged|no change|stay(?:s|ed)? the same"
        r"|remain(?:s|ed)? the same|did not change|didn't change)\b")),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NEGATORS = {"not", "never", "no"}


@dataclass
class Claim:
    sentence: str
    metric: str          # label into _PAIR_PATHS
    phrase: str          # surface phrase that matched
    direction: str       # "up" | "down" | "flat" as claimed
    negated: bool
    actual: str | None   # report's actual direction, None if metric absent
    verdict: str         # "verified" | "contradicted" | "unverifiable"


@dataclass
class ClaimCheckResult:
    claims: list[Claim]
    # coverage: of the sentences that contain direction language (something
    # "went up/down/stayed the same"), how many yielded a checkable claim?
    # A low coverage means the explanation talks about movement in words the
    # lexicon does not recognize -- those sentences are UNCHECKED, and a
    # perfect verified_rate over the extracted claims says nothing about them.
    n_direction_sentences: int = 0
    n_claim_sentences: int = 0

    @property
    def coverage(self) -> float:
        if self.n_direction_sentences == 0:
            return 1.0
        return self.n_claim_sentences / self.n_direction_sentences

    def _by_verdict(self, verdict: str) -> list[Claim]:
        return [c for c in self.claims if c.verdict == verdict]

    @property
    def verified(self) -> list[Claim]:
        return self._by_verdict("verified")

    @property
    def contradicted(self) -> list[Claim]:
        return self._by_verdict("contradicted")

    @property
    def unverifiable(self) -> list[Claim]:
        return self._by_verdict("unverifiable")

    @property
    def verified_rate(self) -> float:
        """Fraction of decidable claims the report supports (1.0 when the
        explanation makes no directional claims, mirroring the convention
        of ``FaithfulnessResult.match_rate``)."""
        decidable = len(self.verified) + len(self.contradicted)
        if decidable == 0:
            return 1.0
        return len(self.verified) / decidable


def _walk(report: dict, path: tuple[str, ...]) -> float | None:
    node = report
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return float(node) if isinstance(node, (int, float)) else None


def _actual_direction(report: dict, metric: str) -> str | None:
    base_path, mod_path = _PAIR_PATHS[metric]
    base, mod = _walk(report, base_path), _walk(report, mod_path)
    if base is None or mod is None:
        return None
    if mod > base + 1e-9:
        return "up"
    if mod < base - 1e-9:
        return "down"
    return "flat"


def _metric_spans(sentence: str) -> list[tuple[int, int, str, str]]:
    """Non-overlapping (start, end, phrase, metric) matches, longest first,
    with trailing generic phrases suppressed near a specific mention."""
    taken: list[tuple[int, int, str, str]] = []
    for phrase, metric in _METRIC_PHRASES:
        for m in re.finditer(re.escape(phrase), sentence):
            span = (m.start(), m.end())
            if any(s < span[1] and span[0] < e for s, e, _, _ in taken):
                continue
            taken.append((span[0], span[1], phrase, metric))
    taken.sort()
    kept: list[tuple[int, int, str, str]] = []
    for start, end, phrase, metric in taken:
        if phrase in _GENERIC_PHRASES and any(
            0 <= start - prev_end <= _GENERIC_SUPPRESS_GAP for _, prev_end, _, _ in kept
        ):
            continue
        kept.append((start, end, phrase, metric))
    return kept


def _is_negated(sentence: str, match_start: int) -> bool:
    tokens = sentence[:match_start].split()[-3:]
    return any(
        t in _NEGATORS or t.endswith("n't") for t in (tok.lower() for tok in tokens)
    )


def extract_claims(explanation: str) -> list[Claim]:
    """Pull (metric, direction) claims out of the text; verdicts unset."""
    claims: list[Claim] = []
    for sentence in _SENTENCE_SPLIT.split(explanation):
        lowered = sentence.lower()
        directions = [
            (m.start(), direction)
            for direction, pattern in _DIRECTION_PATTERNS
            for m in pattern.finditer(lowered)
        ]
        if not directions:
            continue
        for start, _end, phrase, metric in _metric_spans(lowered):
            pos, direction = min(directions, key=lambda d: abs(d[0] - start))
            claims.append(Claim(
                sentence=sentence.strip(), metric=metric, phrase=phrase,
                direction=direction, negated=_is_negated(lowered, pos),
                actual=None, verdict="unverifiable",
            ))
    return claims


def check_claims(report: dict, explanation: str) -> ClaimCheckResult:
    """Extract directional claims from ``explanation`` and verify each one
    against the actual base -> modified movement in ``report``."""
    claims = extract_claims(explanation)
    for claim in claims:
        actual = _actual_direction(report, claim.metric)
        claim.actual = actual
        if actual is None:
            claim.verdict = "unverifiable"
        elif claim.negated:
            claim.verdict = "verified" if actual != claim.direction else "contradicted"
        else:
            claim.verdict = "verified" if actual == claim.direction else "contradicted"

    claim_sentences = {c.sentence for c in claims}
    n_direction, n_claimed = 0, 0
    for sentence in _SENTENCE_SPLIT.split(explanation):
        lowered = sentence.lower()
        if any(p.search(lowered) for _, p in _DIRECTION_PATTERNS):
            n_direction += 1
            if sentence.strip() in claim_sentences:
                n_claimed += 1
    return ClaimCheckResult(
        claims=claims, n_direction_sentences=n_direction, n_claim_sentences=n_claimed,
    )


# ---------------------------------------------------------------------------
# field attribution: is each number attached to the metric it is stated for?
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
# numbers that are part of an identifier ("user 42", "item 900"), not a value
_ID_PREFIX_RE = re.compile(r"\b(?:user|item)\s+$")


@dataclass
class Attribution:
    sentence: str
    metric: str
    phrase: str
    number: float
    verdict: str  # "attributed" | "wrong_field" | "not_in_report"


@dataclass
class AttributionResult:
    attributions: list[Attribution]

    def _by_verdict(self, verdict: str) -> list[Attribution]:
        return [a for a in self.attributions if a.verdict == verdict]

    @property
    def attributed(self) -> list[Attribution]:
        return self._by_verdict("attributed")

    @property
    def wrong_field(self) -> list[Attribution]:
        return self._by_verdict("wrong_field")

    @property
    def not_in_report(self) -> list[Attribution]:
        return self._by_verdict("not_in_report")

    @property
    def attribution_rate(self) -> float:
        if not self.attributions:
            return 1.0
        return len(self.attributed) / len(self.attributions)


def _numeric_leaves(node, out: list[float]) -> None:
    if isinstance(node, dict):
        for v in node.values():
            _numeric_leaves(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _numeric_leaves(v, out)
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        out.append(float(node))


def _subtree_values(report: dict, metric: str) -> list[float]:
    node = report
    for key in _SUBTREE_PATHS[metric]:
        if not isinstance(node, dict) or key not in node:
            return []
        node = node[key]
    out: list[float] = []
    _numeric_leaves(node, out)
    return out


def _close(n: float, values: list[float], tolerance: float) -> bool:
    """Same match rules as check_faithfulness: rounding tolerance, dropped
    sign, and percent/fraction rescaling."""
    return any(
        abs(n - v) <= tolerance
        or abs(n - abs(v)) <= tolerance
        or abs(n - abs(v) * 100) <= tolerance
        for v in values
    )


def check_attribution(
    report: dict, explanation: str, tolerance: float = 0.5
) -> AttributionResult:
    """Verify that each number stated next to a metric phrase matches a value
    under that metric's own report subtree.

    This closes the gap between the presence check (``check_faithfulness``:
    the number exists *somewhere* in the report) and the direction check
    (``check_claims``): "the objective is 3775" passes the presence check
    when 3775 is really the cold-item exposure, but fails here. Numbers with
    no metric phrase in their sentence are not scored (the presence check
    already covers them), as are small integers and identifier numbers like
    "user 42".
    """
    all_values: list[float] = []
    _numeric_leaves(report, all_values)

    attributions: list[Attribution] = []
    for sentence in _SENTENCE_SPLIT.split(explanation):
        lowered = sentence.lower()
        spans = _metric_spans(lowered)
        if not spans:
            continue
        for m in _NUMBER_RE.finditer(lowered):
            # skip digits that are part of a matched metric phrase ("top 10%")
            if any(s <= m.start() < e for s, e, _, _ in spans):
                continue
            if _ID_PREFIX_RE.search(lowered[: m.start()]):
                continue
            n = float(m.group().replace(",", ""))
            # skip small integers ("a couple", "one thing"), but keep small
            # fractions -- gini and share values live below 1
            if abs(n) <= 2 and n == int(n):
                continue
            _start, _end, phrase, metric = min(
                spans, key=lambda s: abs((s[0] + s[1]) / 2 - m.start())
            )
            if _close(n, _subtree_values(report, metric), tolerance):
                verdict = "attributed"
            elif _close(n, all_values, tolerance):
                verdict = "wrong_field"
            else:
                verdict = "not_in_report"
            attributions.append(Attribution(
                sentence=sentence.strip(), metric=metric, phrase=phrase,
                number=n, verdict=verdict,
            ))
    return AttributionResult(attributions=attributions)
