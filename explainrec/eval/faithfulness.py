"""Check that an explanation's numeric claims are grounded in its report.

The explainer is only shown the comparison report (see ``llm/explainer.py``),
so every number it states should trace back to a value in that report. This
module makes that check mechanical instead of assumed: pull every number out
of the explanation text, pull every number out of the report, and flag any
explanation number that has no close match in the report.

This never calls an LLM; it is a pure check over a (report, text) pair, so it
can be unit tested and run against real pipeline output alike.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class FaithfulnessResult:
    claimed: list[float]
    matched: list[float]
    unmatched: list[float]
    unmatched_sentences: list[str] = field(default_factory=list)  # 1:1 with unmatched

    @property
    def match_rate(self) -> float:
        if not self.claimed:
            return 1.0
        return len(self.matched) / len(self.claimed)


def _report_numbers(node, out: set[float]) -> None:
    if isinstance(node, dict):
        for v in node.values():
            _report_numbers(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _report_numbers(v, out)
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        out.add(float(node))


def _extract_report_numbers(report: dict) -> set[float]:
    """All numeric leaves in the report."""
    out: set[float] = set()
    _report_numbers(report, out)
    return out


def _extract_claimed_numbers(text: str) -> list[tuple[float, int]]:
    """(value, character offset) for each number found in ``text``."""
    return [
        (float(m.group().replace(",", "")), m.start())
        for m in _NUMBER_RE.finditer(text)
    ]


def _sentence_at(text: str, offset: int) -> str:
    """The sentence containing character ``offset`` in ``text``."""
    start = 0
    for sentence in _SENTENCE_SPLIT.split(text):
        end = start + len(sentence)
        if start <= offset < end + 2:  # +2 slack for the split whitespace
            return sentence.strip()
        start = end + 1
    return text.strip()


def check_faithfulness(
    report: dict, explanation: str, tolerance: float = 0.5
) -> FaithfulnessResult:
    """Flag numbers in ``explanation`` with no close match in ``report``.

    ``tolerance`` absorbs the LLM's own rounding (e.g. writing "12%" for a
    delta_pct of 12.34, or "about 5" for 5.2); it is intentionally loose
    because this check is meant to catch fabricated numbers, not penalize
    reasonable rounding. Small integers that are near-universal in any
    piece of English (0, 1, 2 — "a couple", "one thing") are excluded from
    scoring so they don't dominate the denominator.
    """
    report_numbers = _extract_report_numbers(report)
    claimed = _extract_claimed_numbers(explanation)

    matched: list[float] = []
    unmatched: list[float] = []
    unmatched_sentences: list[str] = []
    for n, offset in claimed:
        if abs(n) <= 2:
            continue
        # compare against the signed value, its magnitude, and its
        # percent-scaled form: a delta_pct of -2.69 is faithfully restated
        # as either "-2.69%" or "a 2.69% decrease" (sign dropped), and a
        # share stored as a fraction (e.g. 0.658) is faithfully restated
        # as "66%" (scaled by 100); the claim itself may drop the sign or
        # change units without becoming unfaithful.
        if any(
            abs(n - r) <= tolerance
            or abs(n - abs(r)) <= tolerance
            or abs(n - abs(r) * 100) <= tolerance
            for r in report_numbers
        ):
            matched.append(n)
        else:
            unmatched.append(n)
            unmatched_sentences.append(_sentence_at(explanation, offset))

    return FaithfulnessResult(
        claimed=[n for n, _ in claimed if abs(n) > 2],
        matched=matched,
        unmatched=unmatched,
        unmatched_sentences=unmatched_sentences,
    )
