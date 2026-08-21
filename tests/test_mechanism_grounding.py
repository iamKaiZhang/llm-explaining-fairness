from explainrec.constraints import ConstraintSpec, ItemSelector
from explainrec.eval.mechanism_grounding import check_mechanism_grounding, tag_mechanism
from explainrec.scenario import GenderOverride, Modification


def test_tag_constraint_relaxation():
    mod = Modification(summary="s", remove_constraints=["cold-item-exposure"])
    assert tag_mechanism(mod) == "constraint-relaxation"


def test_tag_constraint_addition():
    mod = Modification(
        summary="s",
        add_constraints=[
            ConstraintSpec(name="x", type="forbid_items", items=ItemSelector(kind="cold")),
        ],
    )
    assert tag_mechanism(mod) == "constraint-relaxation"


def test_tag_rating_reestimation():
    mod = Modification(summary="s", gender_overrides=[GenderOverride(user_id=0, gender="F")])
    assert tag_mechanism(mod) == "rating-reestimation"


def test_tag_slate_size_change():
    mod = Modification(summary="s", set_slate_size=5)
    assert tag_mechanism(mod) == "slate-size-change"


def test_tag_noop_is_none():
    mod = Modification(summary="s")
    assert tag_mechanism(mod) is None


def test_tag_multi_mechanism_is_none():
    mod = Modification(
        summary="s",
        remove_constraints=["cold-item-exposure"],
        gender_overrides=[GenderOverride(user_id=0, gender="F")],
    )
    assert tag_mechanism(mod) is None


def test_grounded_explanation_of_constraint_relaxation():
    text = (
        "The system dropped the requirement that every obscure movie reach "
        "5 users, so the constraint no longer limits which items can appear."
    )
    result = check_mechanism_grounding(text, "constraint-relaxation")
    assert result.grounded
    assert result.hits


def test_confabulated_explanation_flagged_as_ungrounded():
    # true mechanism was a constraint being removed, but the explanation
    # invents a rating-model cause instead.
    text = "This happened because the system re-estimated their predicted taste."
    result = check_mechanism_grounding(text, "constraint-relaxation")
    assert not result.grounded
    assert result.misses


def test_grounded_explanation_of_rating_reestimation():
    text = "If this user were female, their predicted rating for these titles changes."
    result = check_mechanism_grounding(text, "rating-reestimation")
    assert result.grounded


def test_wrong_mechanism_vocabulary_flagged_for_rating_reestimation():
    text = "The constraint was removed, which loosened the requirement."
    result = check_mechanism_grounding(text, "rating-reestimation")
    assert not result.grounded


def test_grounded_explanation_of_slate_size_change():
    text = "The slate size was shortened from 10 to 5 recommendations."
    result = check_mechanism_grounding(text, "slate-size-change")
    assert result.grounded


def test_no_expected_terms_is_ungrounded_even_without_misses():
    text = "Numbers went up a bit for reasons that are hard to summarize."
    result = check_mechanism_grounding(text, "constraint-relaxation")
    assert not result.grounded
    assert not result.hits
    assert not result.misses


def test_negated_expected_term_is_not_a_hit():
    text = "This was not caused by any constraint."
    result = check_mechanism_grounding(text, "constraint-relaxation")
    assert not result.hits
    assert not result.grounded


def test_negated_unexpected_term_is_not_a_miss():
    # explicitly ruling OUT the wrong cause is good grounding, not bad
    text = (
        "The requirement was removed; this was not because their taste changed."
    )
    result = check_mechanism_grounding(text, "constraint-relaxation")
    assert result.hits
    assert not result.misses
    assert result.grounded
