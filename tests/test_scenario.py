import pytest

from explainrec.constraints import ConstraintSpec, ItemSelector
from explainrec.ratings import RatingModel
from explainrec.scenario import GenderOverride, Modification, Scenario


@pytest.fixture
def scenario(tiny_dataset):
    model = RatingModel(n_factors=2, n_epochs=3).fit(tiny_dataset)
    cold = ConstraintSpec(
        name="cold-item-exposure", type="min_item_exposure",
        items=ItemSelector(kind="cold"), min_users=1,
    )
    return Scenario(model=model, data=tiny_dataset, constraints=[cold], slate_size=1)


def test_modification_parses_from_json():
    mod = Modification.model_validate({
        "summary": "drop exploration",
        "remove_constraints": ["cold-item-exposure"],
    })
    assert not mod.is_noop()
    assert mod.remove_constraints == ["cold-item-exposure"]


def test_apply_remove_and_add(scenario):
    mod = Modification(
        summary="swap constraint",
        remove_constraints=["cold-item-exposure"],
        add_constraints=[ConstraintSpec(
            name="ban-horror", type="forbid_items",
            items=ItemSelector(kind="genre", genre="Horror"),
        )],
    )
    new = scenario.apply(mod)
    assert [c.name for c in new.constraints] == ["ban-horror"]
    assert [c.name for c in scenario.constraints] == ["cold-item-exposure"]  # unchanged


def test_apply_unknown_removal_raises(scenario):
    mod = Modification(summary="x", remove_constraints=["nope"])
    with pytest.raises(ValueError, match="unknown"):
        scenario.apply(mod)


def test_apply_duplicate_name_raises(scenario):
    mod = Modification(
        summary="x",
        add_constraints=[ConstraintSpec(
            name="cold-item-exposure", type="forbid_items",
            items=ItemSelector(kind="cold"),
        )],
    )
    with pytest.raises(ValueError, match="already in use"):
        scenario.apply(mod)


def test_gender_override_scenario_solves(scenario):
    mod = Modification(
        summary="counterfactual gender for user 0",
        gender_overrides=[GenderOverride(user_id=0, gender="F")],
        focal_users=[0],
    )
    new = scenario.apply(mod)
    assert new.gender_overrides == {0: "F"}
    sol = new.solve()
    assert sol.status == "optimal"


def test_constraint_spec_validation():
    with pytest.raises(ValueError, match="missing"):
        ConstraintSpec(name="bad", type="min_item_exposure")
