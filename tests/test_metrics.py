import numpy as np
import pytest

from explainrec.constraints import ConstraintSpec, ItemSelector
from explainrec.metrics import (
    benefit_inequality, coverage, coverage_disparity, gce, genre_jaccard,
    per_user_metrics,
)
from explainrec.problem import solve_allocation
from explainrec.ratings import RatingModel


def test_coverage(tiny_dataset, tiny_r_hat):
    sol = solve_allocation(tiny_r_hat, [], tiny_dataset, slate_size=1)
    cov = coverage(sol, tiny_dataset)
    # users pick items 0, 1, 2 -> 3 of 4 items recommended
    assert cov == {"fraction": 0.75, "n_recommended": 3, "n_items": 4}


def test_genre_jaccard(tiny_dataset):
    J = genre_jaccard(tiny_dataset)
    # item 0 = [Drama], item 2 = [Drama, Comedy] -> 1/2
    assert J[0, 2] == pytest.approx(0.5)
    # item 0 vs item 1 = [Comedy] -> disjoint
    assert J[0, 1] == 0.0
    assert np.allclose(np.diag(J), 1.0)


def test_coverage_disparity(tiny_dataset, tiny_r_hat):
    sol = solve_allocation(tiny_r_hat, [], tiny_dataset, slate_size=1)
    # exposed: items 0,1,2 with exposure 1 each -> all gaps are 0
    res = coverage_disparity(sol, tiny_dataset, alpha=0.5, epsilon=0.0)
    # similar pairs at alpha=0.5: (0,2) and (1,2); both have zero gap
    assert res["n_similar_pairs"] == 2
    assert res["fraction"] == 1.0

    # force item 3 onto user 0: exposure item0 drops to 0 -> exposed {1,2,3},
    # pair (1,2) similar with exposure 1,1 -> still equal
    forced = solve_allocation(
        tiny_r_hat,
        [ConstraintSpec(name="f", type="force_assign", user_id=0, item_id=3)],
        tiny_dataset, slate_size=1,
    )
    res2 = coverage_disparity(forced, tiny_dataset, alpha=0.4, epsilon=0.0)
    assert res2["n_recommended"] == 3


def test_gce_uniform_is_zero_and_concentration_negative():
    assert gce(np.array([1.0, 1.0, 1.0, 1.0]))["gce"] == pytest.approx(0.0)
    unequal = gce(np.array([1.0, 0.0, 0.0, 0.0]))["gce"]
    assert unequal < -0.5
    # NaNs are excluded, zeros are fine with beta=0.5
    with_nan = gce(np.array([1.0, np.nan, 1.0]))
    assert with_nan == {"gce": pytest.approx(0.0), "n_users": 2}
    with pytest.raises(ValueError):
        gce(np.array([1.0]), beta=1.0)


def test_per_user_metrics(tiny_dataset):
    model = RatingModel(n_factors=2, n_epochs=3).fit(tiny_dataset)
    m = per_user_metrics(model, tiny_dataset, k=2)
    for name in ("MAE", "RMSE", "Precision"):
        assert not np.isnan(m.values[name]).any()
    assert (m.values["MAE"] >= 0).all()
    assert (m.values["RMSE"] >= m.values["MAE"] - 1e-9).all()
    assert ((m.values["Precision"] >= 0) & (m.values["Precision"] <= 1)).all()
    # user 2 has ratings [1,3,5]: one relevant item -> recall/ndcg defined
    assert not np.isnan(m.values["Recall"][2])
    assert 0 <= m.values["nDCG"][2] <= 1


def test_benefit_inequality(tiny_dataset):
    model = RatingModel(n_factors=2, n_epochs=3).fit(tiny_dataset)
    res = benefit_inequality(model, tiny_dataset, "MAE")
    assert res["metric"] == "MAE"
    assert res["gce"] <= 0
    with pytest.raises(ValueError, match="unknown metric"):
        benefit_inequality(model, tiny_dataset, "F1")


def test_item_and_user_groups(tiny_dataset):
    from explainrec.metrics import item_group, user_group

    group, comp = item_group(tiny_dataset, {"kind": "genre", "genre": "Drama"})
    assert group.tolist() == [0, 2] and comp.tolist() == [1, 3]
    cold, warm = item_group(tiny_dataset, {"kind": "cold"})
    assert cold.tolist() == [3] and warm.tolist() == [0, 1, 2]

    assert user_group(tiny_dataset, {"attribute": "gender", "value": "M"}).tolist() == [0, 2]
    assert user_group(tiny_dataset, {"attribute": "age", "min": 0, "max": 29}).tolist() == [0]
    assert user_group(tiny_dataset, {"attribute": "occupation", "value": "b"}).tolist() == [1]
    with pytest.raises(ValueError, match="no users"):
        user_group(tiny_dataset, {"attribute": "occupation", "value": "astronaut"})


def test_group_coverage_gap(tiny_dataset, tiny_r_hat):
    from explainrec.metrics import group_coverage_gap

    sol = solve_allocation(tiny_r_hat, [], tiny_dataset, slate_size=1)
    # exposed {0,1,2}: Drama group {0,2} fully covered; complement {1,3} half
    res = group_coverage_gap(sol, tiny_dataset, {"kind": "genre", "genre": "Drama"})
    assert res["group_fraction"] == 1.0
    assert res["complement_fraction"] == 0.5
    assert res["gap"] == pytest.approx(0.5)


def test_demographic_parity_and_worst_group(tiny_dataset, tiny_r_hat):
    from explainrec.metrics import demographic_parity, worst_group

    sol = solve_allocation(tiny_r_hat, [], tiny_dataset, slate_size=1)
    # slates: u0 -> item0 (5.0), u1 -> item1 (5.0), u2 -> item2 (5.0)
    males = {"attribute": "gender", "value": "M"}
    females = {"attribute": "gender", "value": "F"}
    res = demographic_parity(sol, tiny_r_hat, tiny_dataset, males, females,
                             k=1, stars=4.0, bound=0.05)
    assert res["proportion_x"] == 1.0 and res["proportion_y"] == 1.0
    assert res["gap"] == 0.0 and res["parity_holds"] is True

    wg = worst_group(sol, tiny_r_hat, tiny_dataset,
                     [("men", males), ("women", females)],
                     k=1, utility="mean_predicted_rating", bound=0.1)
    assert wg["per_group"] == {"men": 5.0, "women": 5.0}
    assert wg["gap"] == 0.0 and wg["within_bound"] is True

    # force user 0 onto the cold item -> men's utility drops below women's
    forced = solve_allocation(
        tiny_r_hat,
        [ConstraintSpec(name="f", type="force_assign", user_id=0, item_id=3)],
        tiny_dataset, slate_size=1,
    )
    wg2 = worst_group(forced, tiny_r_hat, tiny_dataset,
                      [("men", males), ("women", females)],
                      k=1, utility="mean_predicted_rating", bound=0.1)
    assert wg2["worst_group"] == "men" and wg2["best_group"] == "women"
    assert wg2["gap"] == pytest.approx(2.0)  # men (5+1)/2=3 vs women 5
    assert wg2["within_bound"] is False
