import pytest

from explainrec.eval.measurement import GradeResult, grade_answer, iter_questions

SET = {
    "set_id": "coverage-disparity-v1",
    "expected_tool": "coverage_disparity",
    "questions": [
        {"id": "cd-01", "question": "main form?", "paraphrases": ["variant?"],
         "answer": 0.8873, "tolerance": 0.01},
        {"id": "bi-01", "question": "inequality?", "paraphrases": [],
         "answer": -0.025, "tolerance": 0.002, "expected_metric": "MAE"},
    ],
}


def test_value_within_tolerance():
    r = grade_answer(SET, "cd-01", 0.89, tool="coverage_disparity")
    assert r.value_correct and r.tool_correct and r.metric_correct is None
    assert r.correct


def test_value_outside_tolerance():
    r = grade_answer(SET, "cd-01", 0.95)
    assert not r.value_correct and not r.correct
    assert r.tool_correct is None  # not reported -> not graded


def test_wrong_tool_fails():
    r = grade_answer(SET, "cd-01", 0.8873, tool="coverage")
    assert r.value_correct and r.tool_correct is False and not r.correct


def test_metric_choice_graded_when_required():
    ok = grade_answer(SET, "bi-01", -0.0251, metric="MAE")
    assert ok.metric_correct and ok.correct
    wrong = grade_answer(SET, "bi-01", -0.0251, metric="RMSE")
    assert wrong.metric_correct is False and not wrong.correct
    unreported = grade_answer(SET, "bi-01", -0.0251)
    assert unreported.metric_correct is False  # required but not given


def test_unknown_question_raises():
    with pytest.raises(KeyError):
        grade_answer(SET, "nope", 0.0)


WG_SET = {
    "set_id": "worst-group-v1",
    "expected_tool": "worst_group",
    "questions": [{
        "id": "wg-01", "question": "who is worst?", "paraphrases": [],
        "answer": 0.3643, "tolerance": 0.02,
        "expected_choice": "female users", "answer_bool": False,
    }],
}


def test_choice_and_bool_grading():
    ok = grade_answer(WG_SET, "wg-01", 0.36, tool="worst_group",
                      choice="Female users", answer_bool=False)
    assert ok.choice_correct and ok.bool_correct and ok.correct

    wrong_choice = grade_answer(WG_SET, "wg-01", 0.36, choice="male users",
                                answer_bool=False)
    assert wrong_choice.choice_correct is False and not wrong_choice.correct

    wrong_bool = grade_answer(WG_SET, "wg-01", 0.36, choice="female users",
                              answer_bool=True)
    assert wrong_bool.bool_correct is False and not wrong_bool.correct

    # sets without categorical/boolean gold don't grade them
    plain = grade_answer(SET, "cd-01", 0.8873, choice="anything", answer_bool=True)
    assert plain.choice_correct is None and plain.bool_correct is None


def test_iter_questions_includes_paraphrases_with_same_gold():
    forms = list(iter_questions(SET))
    assert len(forms) == 3  # 2 mains + 1 paraphrase
    q, text = forms[1]
    assert text == "variant?" and q["id"] == "cd-01"


def test_real_dataset_loads_and_grades():
    from explainrec.eval.measurement import load_set

    try:
        cov = load_set("coverage")
    except FileNotFoundError:
        pytest.skip("dataset not generated")
    q = cov["questions"][0]
    r = grade_answer(cov, q["id"], q["answer"], tool="coverage")
    assert isinstance(r, GradeResult) and r.correct
