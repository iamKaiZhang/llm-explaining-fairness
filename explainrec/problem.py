"""The allocation LP and its solution.

    max  sum_{u,i} r_hat[u,i] * x[u,i]
    s.t. sum_i x[u,i] = slate_size   for every user u
         0 <= x[u,i] <= 1
         + declarative ConstraintSpecs

x is the LP relaxation of the recommend/don't-recommend indicator. The
base constraint matrix is the incidence structure of a bipartite graph
(totally unimodular), so the relaxation typically returns an integral
vertex; ``Solution.n_fractional`` reports when added constraints break
that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .constraints import ConstraintSpec, build_constraint
from .data import Dataset


@dataclass
class Solution:
    x: np.ndarray                # (n_users, n_items) allocation
    objective: float
    status: str
    slate_size: int
    recs: list[np.ndarray]       # per user: item indices sorted by allocation
    exposure: np.ndarray         # per item: expected number of users reached
    n_fractional: int
    solve_seconds: float

    @property
    def mean_predicted_rating(self) -> float | None:
        return getattr(self, "_mean_rating", None)


def solve_allocation(
    r_hat: np.ndarray,
    specs: list[ConstraintSpec],
    data: Dataset,
    slate_size: int = 10,
) -> Solution:
    n_users, n_items = r_hat.shape
    X = cp.Variable((n_users, n_items), nonneg=True)
    constraints: list[cp.Constraint] = [cp.sum(X, axis=1) == slate_size, X <= 1]
    for spec in specs:
        constraints.extend(build_constraint(spec, X, data, slate_size))

    problem = cp.Problem(cp.Maximize(cp.sum(cp.multiply(r_hat, X))), constraints)
    t0 = time.time()
    problem.solve(solver=cp.HIGHS)
    elapsed = time.time() - t0
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"solver status {problem.status!r}: problem is likely infeasible")

    x = np.asarray(X.value)
    x[x < 1e-9] = 0.0
    recs = [np.argsort(-x[u])[:slate_size][x[u, np.argsort(-x[u])[:slate_size]] > 0]
            for u in range(n_users)]
    n_frac = int(((x > 1e-6) & (x < 1 - 1e-6)).sum())

    sol = Solution(
        x=x,
        objective=float(problem.value),
        status=problem.status,
        slate_size=slate_size,
        recs=recs,
        exposure=x.sum(axis=0),
        n_fractional=n_frac,
        solve_seconds=elapsed,
    )
    sol._mean_rating = float((r_hat * x).sum() / x.sum())
    return sol
