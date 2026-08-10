"""End-to-end pipeline: load -> fit -> baseline -> ask.

``Pipeline.ask`` runs the full loop for one natural-language query:
interpret (LLM) -> apply -> re-solve -> compare -> explain (LLM). The
deterministic middle (``run_modification``) is exposed separately so
experiments can bypass the LLM and inject a ``Modification`` directly.
"""

from __future__ import annotations

import hashlib
import json
import pickle
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
    def __init__(
        self,
        baseline: Scenario,
        base_solution: Solution | None = None,
        cache_dir: Path | None = None,
    ):
        self.baseline = baseline
        self._base_solution = base_solution
        self.cache_dir = cache_dir

    @classmethod
    def build(
        cls,
        data_dir: Path = DEFAULT_DATA_DIR,
        slate_size: int = 10,
        constraints: list[ConstraintSpec] | None = None,
        use_cache: bool = True,
    ) -> "Pipeline":
        data = load_dataset(data_dir)
        model = RatingModel().fit(data)
        baseline = Scenario(
            model=model,
            data=data,
            constraints=list(constraints if constraints is not None else DEFAULT_CONSTRAINTS),
            slate_size=slate_size,
        )
        return cls(baseline, cache_dir=data_dir / "cache" if use_cache else None)

    # --- solution caching ------------------------------------------------
    # The model fit is deterministic (seeded), so a scenario's solution is
    # fully determined by the dataset, the model hyperparameters, and the
    # problem definition; that is exactly what the cache key hashes. This
    # applies to the baseline and to every modified scenario alike.

    def _cache_path(self, scenario: Scenario) -> Path:
        m = scenario.model
        key = {
            # bump when the Solution object or solver semantics change,
            # so outdated pickles are ignored
            "solution_version": 2,
            "constraints": [c.model_dump() for c in scenario.constraints],
            "slate_size": scenario.slate_size,
            "gender_overrides": scenario.gender_overrides,
            "n_ratings": len(scenario.data.ratings),
            "cold_threshold": scenario.data.cold_threshold,
            "model": [m.n_factors, m.bias_reg, m.gender_reg, m.mf_reg,
                      m.n_epochs, m.seed],
        }
        digest = hashlib.sha256(
            json.dumps(key, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        return self.cache_dir / f"solution-{digest}.pkl"

    def solve_cached(self, scenario: Scenario) -> Solution:
        """Solve a scenario, fetching from / writing to the disk cache."""
        path = self._cache_path(scenario) if self.cache_dir else None
        if path is not None and path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        solution = scenario.solve()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(solution, f)
        return solution

    @property
    def base_solution(self) -> Solution:
        if self._base_solution is None:
            self._base_solution = self.solve_cached(self.baseline)
        return self._base_solution

    def run_modification(self, mod: Modification) -> dict:
        """Deterministic core: apply a modification, re-solve, compare.

        Raises on invalid or unsolvable modifications; ``ask`` catches and
        explains, experiment code gets the raw error.
        """
        n_users = self.baseline.data.n_users
        bad_focal = [u for u in mod.focal_users if not 0 <= u < n_users]
        if bad_focal:  # validate before spending ~40 s on the solve
            raise ValueError(
                f"focal user ids out of range (0..{n_users - 1}): {bad_focal}"
            )
        modified = self.baseline.apply(mod)
        solution = self.solve_cached(modified)
        return compare_solutions(
            self.base_solution, solution, self.baseline.data,
            focal_users=mod.focal_users or None,
        )

    def ask(
        self, query: str, backend: Backend | None = None,
        skip_explanation: bool = False,
    ) -> AskResult:
        """Answer a what-if query. Never raises: every stage (interpret,
        solve, explain) degrades to an informative response instead."""
        backend = backend or ApiBackend()

        # stage 1: interpret
        try:
            mod = interpret(query, self.baseline, backend)
        except Exception as e:
            mod = Modification(summary="(the query could not be interpreted)")
            return AskResult(query, mod, {"interpretation_failed": str(e)}, (
                "I could not translate this question into a modification of "
                f"the optimization problem ({e}). Try rephrasing it in terms "
                "of what should change: a constraint to add or drop, items "
                "to hide or promote, or a user attribute to flip."
            ))
        if mod.is_noop() and not mod.focal_users:
            return AskResult(query, mod, {}, mod.summary)

        # stage 2: apply + solve + compare
        try:
            report = self.run_modification(mod)
            detail = report_text(report)
        except Exception as e:
            # infeasible problem, solver failure, or an invalid edit that
            # slipped through the schema; explain instead of failing.
            report = {"problem_not_solvable": str(e)}
            detail = (
                f"The requested change could not be solved: {e}\n"
                f"Change: {mod.summary}\n"
                f"Constraints that were active before the change:\n"
                f"{self.baseline.describe_constraints()}\n"
                f"Explain what went wrong (e.g. a conflict between the "
                f"request and these constraints) and suggest how to adjust "
                f"the request; there is no comparison to report."
            )

        # stage 3: explain (fall back to the raw detail if the LLM call fails)
        explanation = None
        if not skip_explanation:
            try:
                explanation = explain(query, mod, detail, backend)
            except Exception as e:
                explanation = (
                    f"(automatic explanation unavailable: {e})\n\n{detail}"
                )
        return AskResult(query, mod, report, explanation)
