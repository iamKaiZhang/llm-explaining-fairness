"""Compare two LLM backends' explanations of the same comparison report.

Every other check in this package asks "is this one explanation correct".
This script isolates a different variable: hold the report fixed (same
solve, same numbers) and vary only which model writes the explanation of
it, then score each model's explanation with the same mechanical checks
(faithfulness, directional claims, mechanism grounding). Any difference in
score is attributable to the explainer model, not to the underlying
problem or edit, since both models are shown the exact same report text.

Like ``run_mechanism_grounding.py`` this solves real (tiny by default)
scenarios via the pipeline's disk cache and calls the real explainer
twice per query (once per backend), so it is slower than ``projection.py``
and meant to be run deliberately.
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
from ..scenario import Modification
from .claims import check_claims
from .faithfulness import check_faithfulness
from .mechanism_grounding import check_mechanism_grounding, tag_mechanism
from .projection import gold_to_modification

QUERIES_PATH = Path(__file__).resolve().parent.parent.parent / "experiments" / "queries.yaml"


@dataclass
class BackendRunResult:
    backend: str
    explanation: str
    match_rate: float
    unmatched: list[float]
    verified_rate: float
    contradicted: list[str]
    mechanism: str | None
    grounded: bool | None
    misses: list[str]


@dataclass
class ComparisonResult:
    id: str
    query: str
    runs: dict[str, BackendRunResult]  # keyed by backend label


def _run_one_backend(
    query: str, mod: Modification, detail: str, report: dict,
    backend: Backend, label: str,
) -> BackendRunResult:
    explanation = explain(query, mod, detail, backend)

    f = check_faithfulness(report, explanation)
    c = check_claims(report, explanation)

    mechanism = tag_mechanism(mod)
    grounded = None
    misses: list[str] = []
    if mechanism is not None:
        m = check_mechanism_grounding(explanation, mechanism)
        grounded = m.grounded
        misses = m.misses

    return BackendRunResult(
        backend=label, explanation=explanation,
        match_rate=round(f.match_rate, 3), unmatched=f.unmatched,
        verified_rate=round(c.verified_rate, 3),
        contradicted=[cl.sentence for cl in c.contradicted],
        mechanism=mechanism, grounded=grounded, misses=misses,
    )


def run(
    backends: dict[str, Backend],
    queries_path: Path = QUERIES_PATH,
    pipeline: Pipeline | None = None,
) -> list[ComparisonResult]:
    """Run every gold-labeled, non-noop query in ``queries_path`` through
    each of ``backends`` (label -> Backend) and score each explanation.

    ``backends`` should have at least two entries to make a comparison
    meaningful, but any number works.
    """
    entries = yaml.safe_load(queries_path.read_text())
    pipeline = pipeline or Pipeline.build()

    results = []
    for entry in entries:
        if not entry.get("gold"):
            continue
        mod = gold_to_modification(entry["gold"])
        if mod.is_noop():
            continue  # nothing to explain

        report = pipeline.run_modification(mod)
        detail = report_text(report)

        runs = {
            label: _run_one_backend(entry["query"], mod, detail, report, backend, label)
            for label, backend in backends.items()
        }
        results.append(ComparisonResult(id=entry["id"], query=entry["query"], runs=runs))
    return results


def summarize(results: list[ComparisonResult]) -> dict:
    labels = sorted({label for r in results for label in r.runs})

    def avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    per_backend = {}
    for label in labels:
        runs = [r.runs[label] for r in results if label in r.runs]
        per_backend[label] = {
            "n_queries": len(runs),
            "avg_faithfulness_match_rate": avg([run.match_rate for run in runs]),
            "avg_claim_verified_rate": avg([run.verified_rate for run in runs]),
            "mechanism_grounded_rate": avg([
                float(run.grounded) for run in runs if run.grounded is not None
            ]),
            "total_unmatched_numbers": sum(len(run.unmatched) for run in runs),
            "total_contradicted_claims": sum(len(run.contradicted) for run in runs),
        }

    return {
        "per_backend": per_backend,
        "disagreements": [
            {
                "id": r.id, "query": r.query,
                "match_rate_by_backend": {l: r.runs[l].match_rate for l in r.runs},
                "verified_rate_by_backend": {l: r.runs[l].verified_rate for l in r.runs},
            }
            for r in results
            if len({r.runs[l].match_rate for l in r.runs}) > 1
            or len({r.runs[l].verified_rate for l in r.runs}) > 1
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", nargs="+", default=["api", "gemini"],
        help="backend names to compare (e.g. api cli gemini)",
    )
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    args = parser.parse_args()

    backends = {name: get_backend(name) for name in args.backends}
    results = run(backends, args.queries)
    print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()
