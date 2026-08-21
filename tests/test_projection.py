from explainrec.constraints import ConstraintSpec, ItemSelector
from explainrec.eval.projection import score_query, summarize
from explainrec.scenario import GenderOverride, Modification, Scenario


class FakeBackend:
    """Returns a fixed Modification regardless of the prompt, for testing
    the scoring logic in isolation from the actual interpreter LLM call."""

    def __init__(self, modification: Modification):
        self._modification = modification

    def structured(self, system, user, schema):
        return self._modification

    def text(self, system, user):
        raise NotImplementedError


def _scenario(tiny_dataset) -> Scenario:
    return Scenario(model=None, data=tiny_dataset, constraints=[], slate_size=2)


def test_exact_match_on_gender_override(tiny_dataset):
    entry = {
        "id": "cf-1", "role": "end_user", "type": "counterfactual",
        "query": "would user 0 get different movies as a woman?",
        "gold": {"gender_overrides": [{"user_id": 0, "gender": "F"}], "focal_users": [0]},
    }
    predicted = Modification(
        summary="flip gender",
        gender_overrides=[GenderOverride(user_id=0, gender="F")],
        focal_users=[0],
    )
    result = score_query(entry, _scenario(tiny_dataset), FakeBackend(predicted))
    assert result.correct
    assert result.field_matches["gender_overrides"]
    assert result.field_matches["focal_users"]


def test_wrong_field_marks_incorrect(tiny_dataset):
    entry = {
        "id": "cf-2", "role": "end_user", "type": "counterfactual",
        "query": "would user 0 get different movies as a woman?",
        "gold": {"gender_overrides": [{"user_id": 0, "gender": "F"}], "focal_users": [0]},
    }
    # interpreter picked the wrong user
    predicted = Modification(
        summary="flip gender",
        gender_overrides=[GenderOverride(user_id=1, gender="F")],
        focal_users=[1],
    )
    result = score_query(entry, _scenario(tiny_dataset), FakeBackend(predicted))
    assert not result.correct
    assert not result.field_matches["gender_overrides"]


def test_inexpressible_query_correct_when_noop(tiny_dataset):
    entry = {
        "id": "reg-1", "role": "regulator", "type": "evaluative",
        "query": "make sure no group gets worse slates",
        "gold": None,
    }
    predicted = Modification(summary="no change possible")
    result = score_query(entry, _scenario(tiny_dataset), FakeBackend(predicted))
    assert result.correct
    assert not result.expressible


def test_inexpressible_query_flagged_as_false_projection(tiny_dataset):
    entry = {
        "id": "reg-2", "role": "regulator", "type": "evaluative",
        "query": "make sure no group gets worse slates",
        "gold": None,
    }
    # interpreter invented an edit for a request the schema can't represent
    predicted = Modification(
        summary="hides cold items",
        add_constraints=[
            ConstraintSpec(name="x", type="forbid_items", items=ItemSelector(kind="cold")),
        ],
    )
    result = score_query(entry, _scenario(tiny_dataset), FakeBackend(predicted))
    assert not result.correct
    assert "false projection" in result.note


def test_constraint_order_does_not_affect_match(tiny_dataset):
    entry = {
        "id": "op-1", "role": "operator", "type": "interventional",
        "query": "cap item 0 at 5 users and forbid item 1 for everyone",
        "gold": {
            "add_constraints": [
                {"name": "cap-0", "type": "max_item_exposure",
                 "items": {"kind": "ids", "ids": [0]}, "max_users": 5},
                {"name": "forbid-1", "type": "forbid_items",
                 "items": {"kind": "ids", "ids": [1]}},
            ],
        },
    }
    # same two constraints, opposite order
    predicted = Modification(
        summary="two edits",
        add_constraints=[
            ConstraintSpec(name="forbid-1", type="forbid_items",
                            items=ItemSelector(kind="ids", ids=[1])),
            ConstraintSpec(name="cap-0", type="max_item_exposure",
                            items=ItemSelector(kind="ids", ids=[0]), max_users=5),
        ],
    )
    result = score_query(entry, _scenario(tiny_dataset), FakeBackend(predicted))
    assert result.correct


def test_summarize_reports_rates_by_role_and_type(tiny_dataset):
    scenario = _scenario(tiny_dataset)
    correct_entry = {
        "id": "a", "role": "end_user", "type": "counterfactual",
        "query": "q", "gold": {"focal_users": [0]},
    }
    wrong_entry = {
        "id": "b", "role": "operator", "type": "interventional",
        "query": "q", "gold": {"focal_users": [0]},
    }
    results = [
        score_query(correct_entry, scenario, FakeBackend(Modification(summary="s", focal_users=[0]))),
        score_query(wrong_entry, scenario, FakeBackend(Modification(summary="s", focal_users=[1]))),
    ]
    summary = summarize(results)
    assert summary["by_role"]["end_user"] == 1.0
    assert summary["by_role"]["operator"] == 0.0
    assert summary["overall_accuracy"] == 0.5
    assert len(summary["failures"]) == 1
