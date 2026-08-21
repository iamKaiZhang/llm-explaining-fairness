"""Score the LLM interpreter against the gold modifications in queries.yaml.

For each query with a gold ``Modification``, run the interpreter and diff
the fields. For each query with ``gold: null`` (the schema cannot express
the request), the correct behavior is a no-op-like modification; anything
else is a false projection onto a nearby expressible edit.

This calls the LLM interpreter but never solves the optimization problem,
so it is cheap to run repeatedly while iterating on the interpreter prompt.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..llm.backend import Backend, get_backend
from ..llm.interpreter import interpret
from ..pipeline import DEFAULT_CONSTRAINTS
from ..scenario import GenderOverride, Modification, Scenario

QUERIES_PATH = Path(__file__).resolve().parent.parent.parent / "experiments" / "queries.yaml"


@dataclass
class QueryResult:
    id: str
    role: str
    type: str
    expressible: bool
    query: str
    gold: dict | None
    predicted: dict
    field_matches: dict[str, bool] = field(default_factory=dict)
    correct: bool = False
    note: str = ""


def gold_to_modification(gold: dict) -> Modification:
    gold = dict(gold)
    gender_overrides = [GenderOverride(**g) for g in gold.pop("gender_overrides", [])]
    return Modification(
        summary="(gold)",
        add_constraints=gold.pop("add_constraints", []),
        remove_constraints=gold.pop("remove_constraints", []),
        gender_overrides=gender_overrides,
        set_slate_size=gold.pop("set_slate_size", None),
        focal_users=gold.pop("focal_users", []),
    )


_COMPARE_FIELDS = [
    "add_constraints", "remove_constraints", "gender_overrides",
    "set_slate_size", "focal_users",
]


def _normalize(mod: Modification) -> dict:
    dumped = mod.model_dump(exclude={"summary"})
    dumped["add_constraints"] = sorted(
        (json.dumps(c, sort_keys=True) for c in dumped["add_constraints"]),
    )
    dumped["remove_constraints"] = sorted(dumped["remove_constraints"])
    dumped["gender_overrides"] = sorted(
        (json.dumps(g, sort_keys=True) for g in dumped["gender_overrides"]),
    )
    dumped["focal_users"] = sorted(dumped["focal_users"])
    return dumped


def score_modification(entry: dict, predicted: Modification) -> QueryResult:
    """Score an already-produced Modification against ``entry``'s gold.

    Split out from ``score_query`` so callers that already interpreted a
    query (e.g. the web demo, after ``Pipeline.ask``) can reuse that result
    instead of paying for a second, separate interpret() call.
    """
    predicted_norm = _normalize(predicted)

    if entry["gold"] is None:
        # correct behavior: decline, i.e. emit (close to) a no-op.
        is_false_projection = not predicted.is_noop()
        return QueryResult(
            id=entry["id"], role=entry["role"], type=entry["type"],
            expressible=False, query=entry["query"], gold=None,
            predicted=predicted_norm,
            correct=not is_false_projection,
            note="false projection: emitted a non-noop edit for an inexpressible request"
                 if is_false_projection else "correctly emitted a no-op",
        )

    gold_mod = gold_to_modification(entry["gold"])
    gold_norm = _normalize(gold_mod)
    field_matches = {f: predicted_norm[f] == gold_norm[f] for f in _COMPARE_FIELDS}
    return QueryResult(
        id=entry["id"], role=entry["role"], type=entry["type"],
        expressible=True, query=entry["query"], gold=gold_norm,
        predicted=predicted_norm, field_matches=field_matches,
        correct=all(field_matches.values()),
    )


def score_query(entry: dict, scenario: Scenario, backend: Backend) -> QueryResult:
    predicted = interpret(entry["query"], scenario, backend)
    return score_modification(entry, predicted)


def find_entry_by_query(
    query: str, queries_path: Path = QUERIES_PATH,
) -> dict | None:
    """Look up a manifest entry with exactly this query text, if any."""
    entries = yaml.safe_load(queries_path.read_text())
    query_norm = query.strip().lower()
    for entry in entries:
        if entry["query"].strip().lower() == query_norm:
            return entry
    return None


def run(
    queries_path: Path = QUERIES_PATH, backend_name: str = "api",
) -> list[QueryResult]:
    entries = yaml.safe_load(queries_path.read_text())
    from ..data import DEFAULT_DATA_DIR, load_dataset

    # interpret() only reads scenario.data / .slate_size / .describe_constraints(),
    # so the (expensive) fitted rating model is not needed here.
    scenario = Scenario(
        model=None,
        data=load_dataset(DEFAULT_DATA_DIR),
        constraints=list(DEFAULT_CONSTRAINTS),
        slate_size=10,
    )

    backend = get_backend(backend_name)
    return [score_query(entry, scenario, backend) for entry in entries]


def summarize(results: list[QueryResult]) -> dict:
    def rate(subset: list[QueryResult]) -> float | None:
        return round(sum(r.correct for r in subset) / len(subset), 3) if subset else None

    by_role, by_type = {}, {}
    for r in results:
        by_role.setdefault(r.role, []).append(r)
        by_type.setdefault(r.type, []).append(r)

    expressible = [r for r in results if r.expressible]
    inexpressible = [r for r in results if not r.expressible]

    return {
        "overall_accuracy": rate(results),
        "expressible_accuracy": rate(expressible),
        "false_projection_rate": (
            round(1 - rate(inexpressible), 3) if inexpressible else None
        ),
        "by_role": {k: rate(v) for k, v in by_role.items()},
        "by_type": {k: rate(v) for k, v in by_type.items()},
        "failures": [
            {"id": r.id, "query": r.query, "field_matches": r.field_matches, "note": r.note}
            for r in results if not r.correct
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["api", "cli"], default="api")
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    args = parser.parse_args()

    results = run(args.queries, args.backend)
    summary = summarize(results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
