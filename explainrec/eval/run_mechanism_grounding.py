"""Run the mechanism-grounding check over queries.yaml's gold modifications.

For each entry with a gold ``Modification`` that exercises exactly one
mechanism (see ``tag_mechanism``), apply it to the baseline scenario, solve
and compare (using the pipeline's disk cache, same as production), explain
the report with the real explainer, and check the explanation's causal
language against that mechanism's rubric.

This solves real (tiny by default) scenarios and calls both LLM stages
(interpreter is bypassed -- gold is used directly -- but the explainer is
real), so it is slower than ``projection.py`` and meant to be run
deliberately, not on every prompt change.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..compare import report_text
from ..llm.backend import Backend, get_backend
from ..llm.explainer import explain
from ..pipeline import Pipeline
from .mechanism_grounding import check_mechanism_grounding, tag_mechanism
from .projection import gold_to_modification

QUERIES_PATH = Path(__file__).resolve().parent.parent.parent / "experiments" / "queries.yaml"


@dataclass
class MechanismRunResult:
    id: str
    mechanism: str
    query: str
    explanation: str
    hits: list[str]
    misses: list[str]
    grounded: bool


def run(
    queries_path: Path = QUERIES_PATH,
    backend_name: str = "api",
    pipeline: Pipeline | None = None,
    include_change_summary: bool = True,
) -> list[MechanismRunResult]:
    entries = yaml.safe_load(queries_path.read_text())
    pipeline = pipeline or Pipeline.build()
    backend: Backend = get_backend(backend_name)

    results = []
    for entry in entries:
        if not entry.get("gold"):
            continue
        mod = gold_to_modification(entry["gold"])
        mechanism = tag_mechanism(mod)
        if mechanism is None:
            continue  # no-op or multi-mechanism edit: no single ground truth

        report = pipeline.run_modification(mod)
        detail = report_text(report)
        explanation = explain(
            entry["query"], mod, detail, backend,
            include_change_summary=include_change_summary,
        )
        check = check_mechanism_grounding(explanation, mechanism)

        results.append(MechanismRunResult(
            id=entry["id"], mechanism=mechanism, query=entry["query"],
            explanation=explanation, hits=check.hits, misses=check.misses,
            grounded=check.grounded,
        ))
    return results


def summarize(results: list[MechanismRunResult]) -> dict:
    def rate(subset: list[MechanismRunResult]) -> float | None:
        return round(sum(r.grounded for r in subset) / len(subset), 3) if subset else None

    by_mechanism: dict[str, list[MechanismRunResult]] = {}
    for r in results:
        by_mechanism.setdefault(r.mechanism, []).append(r)

    return {
        "overall_grounded_rate": rate(results),
        "by_mechanism": {k: rate(v) for k, v in by_mechanism.items()},
        "ungrounded": [
            {"id": r.id, "mechanism": r.mechanism, "misses": r.misses, "hits": r.hits}
            for r in results if not r.grounded
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["api", "cli"], default="api")
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument(
        "--no-summary", action="store_true",
        help="withhold the applied change from the explainer prompt, so the "
             "mechanism must be inferred from the report (the ablation run)",
    )
    args = parser.parse_args()

    results = run(
        args.queries, args.backend, include_change_summary=not args.no_summary,
    )
    summary = summarize(results)
    summary["change_summary_included"] = not args.no_summary
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
