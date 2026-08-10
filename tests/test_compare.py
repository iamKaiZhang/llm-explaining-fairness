from explainrec.compare import compare_solutions, report_text
from explainrec.constraints import ConstraintSpec
from explainrec.problem import solve_allocation


def test_focal_diff_tracks_cold_items(tiny_dataset, tiny_r_hat):
    # item 3 is the cold item; force it onto user 0 in the modified solution
    base = solve_allocation(tiny_r_hat, [], tiny_dataset, slate_size=1)
    forced = solve_allocation(
        tiny_r_hat,
        [ConstraintSpec(name="force", type="force_assign", user_id=0, item_id=3)],
        tiny_dataset, slate_size=1,
    )
    report = compare_solutions(base, forced, tiny_dataset, focal_users=[0])

    focal = report["focal_users"]["0"]
    assert focal["cold_items_in_slate"] == {"base": 0, "modified": 1}
    assert focal["added"] == ["D [cold]"]      # cold title is labeled
    assert focal["removed"] == ["A"]           # warm title is not

    exposure = report["cold_item_exposure"]["mean_cold_items_per_user_slate"]
    assert exposure["base"] == 0.0
    assert exposure["modified"] == round(1 / 3, 2)

    # labels survive the text rendering the explainer sees
    assert "D [cold]" in report_text(report)


def test_distributional_metrics(tiny_dataset, tiny_r_hat):
    base = solve_allocation(tiny_r_hat, [], tiny_dataset, slate_size=1)
    forced = solve_allocation(
        tiny_r_hat,
        [ConstraintSpec(name="force", type="force_assign", user_id=0, item_id=3)],
        tiny_dataset, slate_size=1,
    )
    report = compare_solutions(base, forced, tiny_dataset, focal_users=[0])

    # everyone gets their 5.0 item in the base; user 0 drops to 1.0 when forced
    rating = report["per_user_slate_rating_distribution"]
    assert rating["base"]["min"] == 5.0
    assert rating["modified"]["min"] == 1.0

    # cold burden: [0,0,0] -> [1,0,0]; gini reflects the concentration
    burden = report["exploration_burden_distribution"]
    assert burden["base"]["max"] == 0.0
    assert burden["modified"]["max"] == 1.0
    assert burden["base"]["gini"] == 0.0
    assert 0.0 < burden["modified"]["gini"] <= 1.0

    # focal user sees their own predicted rating drop
    focal = report["focal_users"]["0"]
    assert focal["mean_predicted_slate_rating"] == {"base": 5.0, "modified": 1.0}

    # the focal profile carries the user's recorded attributes and history
    profile = focal["profile"]
    assert profile["age"] == 25
    assert profile["gender"] == "M"
    assert profile["occupation"] == "a"
    assert profile["n_ratings"] == 4                    # items 0,1,2,3
    assert profile["mean_rating_given"] == 2.75         # (5+3+1+2)/4

    conc = report["item_exposure_concentration"]
    assert 0.0 <= conc["gini"]["modified"] <= 1.0
    assert 0.0 < conc["top_10pct_items_exposure_share"]["base"] <= 1.0
