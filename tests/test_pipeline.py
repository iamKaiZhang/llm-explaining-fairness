import pytest

from explainrec.constraints import ConstraintSpec, ItemSelector
from explainrec.pipeline import Pipeline
from explainrec.ratings import RatingModel
from explainrec.scenario import Modification, Scenario


@pytest.fixture
def tiny_pipeline(tiny_dataset, tmp_path):
    model = RatingModel(n_factors=2, n_epochs=3).fit(tiny_dataset)
    baseline = Scenario(model=model, data=tiny_dataset, constraints=[], slate_size=1)
    return Pipeline(baseline, cache_dir=tmp_path / "cache")


class StubBackend:
    """Backend returning a fixed Modification; no LLM involved."""

    def __init__(self, mod: Modification):
        self.mod = mod

    def structured(self, system, user, schema):
        return self.mod

    def text(self, system, user):
        return "stub explanation"


def test_base_solution_is_cached_to_disk(tiny_pipeline):
    sol = tiny_pipeline.base_solution
    path = tiny_pipeline._cache_path(tiny_pipeline.baseline)
    assert path.exists()

    # a fresh pipeline over the same problem loads from disk instead of solving
    reloaded = Pipeline(tiny_pipeline.baseline, cache_dir=tiny_pipeline.cache_dir)
    reloaded.baseline = tiny_pipeline.baseline
    reloaded.baseline.solve = None  # would crash if solving were attempted
    assert reloaded.base_solution.objective == pytest.approx(sol.objective)


def test_cache_key_changes_with_problem(tiny_pipeline, tiny_dataset):
    path_before = tiny_pipeline._cache_path(tiny_pipeline.baseline)
    model = tiny_pipeline.baseline.model
    other = Scenario(model=model, data=tiny_dataset,
                     constraints=[], slate_size=2)  # different slate size
    assert tiny_pipeline._cache_path(other) != path_before


def test_modified_solutions_are_cached_too(tiny_pipeline):
    mod = Modification(summary="force item 3 for user 0",
                       add_constraints=[ConstraintSpec(
                           name="force", type="force_assign", user_id=0, item_id=3)])
    report_first = tiny_pipeline.run_modification(mod)
    # baseline + modified scenario -> two cache files
    files = sorted(tiny_pipeline.cache_dir.glob("solution-*.pkl"))
    assert len(files) == 2

    # second run of the same modification loads from disk instead of solving
    from explainrec import scenario as scenario_module
    original_solve = scenario_module.Scenario.solve
    scenario_module.Scenario.solve = None  # would crash if solving were attempted
    try:
        report_second = tiny_pipeline.run_modification(mod)
    finally:
        scenario_module.Scenario.solve = original_solve
    assert report_second == report_first


def test_ask_reports_infeasibility_gracefully(tiny_pipeline):
    # forbidding every item makes the per-user slate constraint unsatisfiable
    mod = Modification(
        summary="ban everything",
        add_constraints=[ConstraintSpec(
            name="ban-all", type="forbid_items", items=ItemSelector(kind="all"),
        )],
    )
    result = tiny_pipeline.ask("what if nothing may be recommended?",
                               backend=StubBackend(mod))
    assert "problem_not_solvable" in result.report
    assert result.explanation == "stub explanation"


def test_ask_normal_path_with_stub_backend(tiny_pipeline):
    mod = Modification(summary="force item 3 for user 0",
                       add_constraints=[ConstraintSpec(
                           name="force", type="force_assign", user_id=0, item_id=3)])
    result = tiny_pipeline.ask("force it", backend=StubBackend(mod))
    assert result.report["objective"]["modified"] <= result.report["objective"]["base"]
    assert result.explanation == "stub explanation"


class BrokenInterpreter(StubBackend):
    def structured(self, system, user, schema):
        raise RuntimeError("backend down")


class BrokenExplainer(StubBackend):
    def text(self, system, user):
        raise RuntimeError("backend down")


def test_ask_survives_interpreter_failure(tiny_pipeline):
    result = tiny_pipeline.ask("anything", backend=BrokenInterpreter(None))
    assert "interpretation_failed" in result.report
    assert "could not translate" in result.explanation


def test_ask_survives_explainer_failure_with_report_fallback(tiny_pipeline):
    mod = Modification(summary="force item 3 for user 0",
                       add_constraints=[ConstraintSpec(
                           name="force", type="force_assign", user_id=0, item_id=3)])
    result = tiny_pipeline.ask("force it", backend=BrokenExplainer(mod))
    # solve succeeded; explanation falls back to the raw report
    assert "objective" in result.report
    assert "explanation unavailable" in result.explanation
    assert "objective" in result.explanation  # report text included


def test_ask_survives_out_of_range_ids_from_llm(tiny_pipeline):
    mod = Modification(summary="force item for a user that does not exist",
                       add_constraints=[ConstraintSpec(
                           name="bad", type="force_assign", user_id=999, item_id=0)])
    result = tiny_pipeline.ask("bad ids", backend=StubBackend(mod))
    assert "problem_not_solvable" in result.report
    assert "out of range" in result.report["problem_not_solvable"]


def test_run_modification_rejects_bad_focal_users_before_solving(tiny_pipeline):
    mod = Modification(summary="focal out of range", focal_users=[999])
    with pytest.raises(ValueError, match="focal user ids out of range"):
        tiny_pipeline.run_modification(mod)
