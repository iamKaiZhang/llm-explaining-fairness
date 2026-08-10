import numpy as np
import pytest

from explainrec.constraints import ConstraintSpec, ItemSelector
from explainrec.problem import solve_allocation


def test_unconstrained_picks_argmax(tiny_dataset, tiny_r_hat):
    sol = solve_allocation(tiny_r_hat, [], tiny_dataset, slate_size=1)
    picks = [recs[0] for recs in sol.recs]
    assert picks == [0, 1, 2]  # each user gets their best item
    assert sol.objective == pytest.approx(15.0)
    assert sol.n_fractional == 0


def test_min_exposure_forces_cold_item(tiny_dataset, tiny_r_hat):
    spec = ConstraintSpec(
        name="cold", type="min_item_exposure",
        items=ItemSelector(kind="cold"), min_users=1,
    )
    sol = solve_allocation(tiny_r_hat, [spec], tiny_dataset, slate_size=1)
    assert sol.exposure[3] >= 1 - 1e-6      # item 3 is the cold item
    # cheapest sacrifice: any user gives up 4 points (5 -> 1)
    assert sol.objective == pytest.approx(11.0)


def test_forbid_items(tiny_dataset, tiny_r_hat):
    spec = ConstraintSpec(
        name="ban-drama", type="forbid_items",
        items=ItemSelector(kind="genre", genre="Drama"),
    )
    sol = solve_allocation(tiny_r_hat, [spec], tiny_dataset, slate_size=1)
    assert sol.exposure[0] == pytest.approx(0.0)
    assert sol.exposure[2] == pytest.approx(0.0)
    picks = [recs[0] for recs in sol.recs]
    assert picks == [1, 1, 1]  # only B and D remain; B dominates


def test_force_assign(tiny_dataset, tiny_r_hat):
    spec = ConstraintSpec(name="force", type="force_assign", user_id=0, item_id=3)
    sol = solve_allocation(tiny_r_hat, [spec], tiny_dataset, slate_size=1)
    assert sol.x[0, 3] == pytest.approx(1.0)


def test_infeasible_exposure_raises(tiny_dataset, tiny_r_hat):
    spec = ConstraintSpec(
        name="too-much", type="min_item_exposure",
        items=ItemSelector(kind="all"), min_users=2,
    )
    with pytest.raises(ValueError, match="infeasible"):
        solve_allocation(tiny_r_hat, [spec], tiny_dataset, slate_size=1)
