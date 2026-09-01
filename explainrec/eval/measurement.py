"""Grade answers against the measurement-query dataset (datasets/).

A system under test answers a measurement query with (a) the tool it chose,
(b) the metric, when the set requires choosing one, and (c) a numeric value.
``grade_answer`` scores all three against the dataset's gold labels; the
numeric check uses each question's own tolerance.

Pure functions over the dataset JSON — no LLM, no solver.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "llm_eval_dataset"


@dataclass
class GradeResult:
    question_id: str
    value_correct: bool
    tool_correct: bool | None    # None: no tool reported / not graded
    metric_correct: bool | None  # None: set does not require a metric choice
    choice_correct: bool | None  # None: set has no categorical answer
    bool_correct: bool | None    # None: set has no yes/no answer
    expected_value: float
    tolerance: float

    @property
    def correct(self) -> bool:
        return (self.value_correct
                and self.tool_correct is not False
                and self.metric_correct is not False
                and self.choice_correct is not False
                and self.bool_correct is not False)


def load_set(name: str, dataset_dir: Path = DATASET_DIR) -> dict:
    return json.loads((dataset_dir / f"{name}.json").read_text())


def iter_questions(question_set: dict):
    """Yield (question_dict, question_text) pairs, paraphrases included.

    Every surface form of a question carries the same gold answer, per the
    dataset design: paraphrasing never changes ground truth.
    """
    for q in question_set["questions"]:
        yield q, q["question"]
        for p in q.get("paraphrases", []):
            yield q, p


def grade_answer(
    question_set: dict,
    question_id: str,
    value: float,
    tool: str | None = None,
    metric: str | None = None,
    choice: str | None = None,
    answer_bool: bool | None = None,
) -> GradeResult:
    """Grade one answer. ``value`` is always graded; ``metric``, ``choice``
    (a categorical answer, e.g. which group is worst served), and
    ``answer_bool`` (a yes/no verdict, e.g. parity holds) are graded exactly
    when the question carries the corresponding gold field."""
    question = next(
        (q for q in question_set["questions"] if q["id"] == question_id), None
    )
    if question is None:
        raise KeyError(f"no question {question_id!r} in {question_set['set_id']}")

    value_correct = abs(value - question["answer"]) <= question["tolerance"]
    tool_correct = None if tool is None else (tool == question_set["expected_tool"])

    expected_metric = question.get("expected_metric")
    metric_correct = None
    if expected_metric is not None:
        metric_correct = metric == expected_metric

    expected_choice = question.get("expected_choice")
    choice_correct = None
    if expected_choice is not None:
        choice_correct = (choice is not None
                          and choice.strip().casefold() == expected_choice.casefold())

    expected_bool = question.get("answer_bool")
    bool_correct = None
    if expected_bool is not None:
        bool_correct = answer_bool == expected_bool

    return GradeResult(
        question_id=question_id,
        value_correct=value_correct,
        tool_correct=tool_correct,
        metric_correct=metric_correct,
        choice_correct=choice_correct,
        bool_correct=bool_correct,
        expected_value=question["answer"],
        tolerance=question["tolerance"],
    )
