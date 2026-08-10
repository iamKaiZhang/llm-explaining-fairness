"""End-to-end pipeline: load -> fit -> baseline -> ask.

``Pipeline.ask`` runs the full loop for one natural-language query:
interpret (LLM) -> apply -> re-solve -> compare -> explain (LLM). The
deterministic middle (``run_modification``) is exposed separately so
experiments can bypass the LLM and inject a ``Modification`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .compare import compare_solutions, report_text
from .constraints import ConstraintSpec, ItemSelector
from .data import DEFAULT_DATA_DIR, Dataset, load_dataset
from .llm.backend import ApiBackend, Backend
from .llm.explainer import explain
from .llm.interpreter import interpret
from .problem import Solution
from .ratings import RatingModel
from .scenario import Modification, Scenario

# The baseline problem: pure rating maximization plus one fairness
# constraint of the kind discussed in the project notes - every cold
# item must reach at least 5 users.
DEFAULT_CONSTRAINTS = [
    ConstraintSpec(
        name="cold-item-exposure",
        type="min_item_exposure",
        items=ItemSelector(kind="cold"),
        min_users=5,
    ),
]


@dataclass
class AskResult:
    query: str
    modification: Modification
    report: dict
    explanation: str | None


class Pipeline:
    def __init__(self, baseline: Scenario, base_solution: Solution | None = None):
        self.baseline = baseline
        self._base_solution = base_solution

    @classmethod
    def build(
        cls,
        data_dir: Path = DEFAULT_DATA_DIR,
        slate_size: int = 10,
        constraints: list[ConstraintSpec] | None = None,
    ) -> "Pipeline":
        data = load_dataset(data_dir)
        model = RatingModel().fit(data)
        baseline = Scenario(
            model=model,
            data=data,
            constraints=list(constraints if constraints is not None else DEFAULT_CONSTRAINTS),
            slate_size=slate_size,
        )
        return cls(baseline)

    @property
    def base_solution(self) -> Solution:
        if self._base_solution is None:
            self._base_solution = self.baseline.solve()
        return self._base_solution

    def run_modification(self, mod: Modification) -> dict:
        """Deterministic core: apply a modification, re-solve, compare."""
        modified = self.baseline.apply(mod)
        solution = modified.solve()
        return compare_solutions(
            self.base_solution, solution, self.baseline.data,
            focal_users=mod.focal_users or None,
        )

    def ask(
        self, query: str, backend: Backend | None = None,
        skip_explanation: bool = False,
    ) -> AskResult:
        backend = backend or ApiBackend()
        mod = interpret(query, self.baseline, backend)
        if mod.is_noop() and not mod.focal_users:
            return AskResult(query, mod, {}, mod.summary)
        report = self.run_modification(mod)
        explanation = None
        if not skip_explanation:
            explanation = explain(query, mod, report_text(report), backend)
        return AskResult(query, mod, report, explanation)
