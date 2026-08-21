from explainrec.eval.compare_backends import (
    BackendRunResult, ComparisonResult, summarize,
)


def _run(backend, match_rate, verified_rate, unmatched=None, contradicted=None,
         grounded=None):
    return BackendRunResult(
        backend=backend, explanation="...", match_rate=match_rate,
        unmatched=unmatched or [], verified_rate=verified_rate,
        contradicted=contradicted or [], mechanism="constraint-relaxation",
        grounded=grounded, misses=[],
    )


def test_per_backend_averages_computed_independently():
    results = [
        ComparisonResult(id="q1", query="Q1", runs={
            "claude": _run("claude", 1.0, 1.0),
            "gemini": _run("gemini", 0.5, 0.5, unmatched=[42.0]),
        }),
        ComparisonResult(id="q2", query="Q2", runs={
            "claude": _run("claude", 1.0, 1.0),
            "gemini": _run("gemini", 1.0, 1.0),
        }),
    ]
    summary = summarize(results)
    assert summary["per_backend"]["claude"]["avg_faithfulness_match_rate"] == 1.0
    assert summary["per_backend"]["gemini"]["avg_faithfulness_match_rate"] == 0.75
    assert summary["per_backend"]["gemini"]["total_unmatched_numbers"] == 1
    assert summary["per_backend"]["claude"]["n_queries"] == 2


def test_disagreements_only_lists_queries_where_scores_differ():
    results = [
        ComparisonResult(id="q1", query="Q1", runs={
            "claude": _run("claude", 1.0, 1.0),
            "gemini": _run("gemini", 1.0, 1.0),
        }),
        ComparisonResult(id="q2", query="Q2", runs={
            "claude": _run("claude", 1.0, 1.0),
            "gemini": _run("gemini", 0.5, 1.0),
        }),
    ]
    summary = summarize(results)
    ids = [d["id"] for d in summary["disagreements"]]
    assert ids == ["q2"]


def test_mechanism_grounded_rate_ignores_none():
    results = [
        ComparisonResult(id="q1", query="Q1", runs={
            "claude": _run("claude", 1.0, 1.0, grounded=True),
            "gemini": _run("gemini", 1.0, 1.0, grounded=False),
        }),
    ]
    summary = summarize(results)
    assert summary["per_backend"]["claude"]["mechanism_grounded_rate"] == 1.0
    assert summary["per_backend"]["gemini"]["mechanism_grounded_rate"] == 0.0


def test_empty_results_produce_empty_summary():
    summary = summarize([])
    assert summary["per_backend"] == {}
    assert summary["disagreements"] == []
