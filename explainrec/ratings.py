"""Rating estimation: biases + gender pathway + matrix factorization.

The prediction is additive:

    r_hat(u, i) = mu + b_u + b_i + delta(gender(u), i) + p_u . q_i

fit in stages: global/user/item biases first, then a shrunk gender-item
effect on the bias residuals, then ALS matrix factorization on what
remains. The explicit ``delta`` pathway is what makes attribute
counterfactuals ("what if user u were male?") well defined: flipping
the attribute swaps only the demographic term.

Caveat (documented on purpose): ``b_u`` and ``p_u`` are learned from
the user's actual rating history, which itself may correlate with
gender. The counterfactual therefore intervenes on the *explicit*
demographic pathway only, not on the history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset

GENDERS = ["M", "F"]


class RatingModel:
    def __init__(
        self,
        n_factors: int = 20,
        bias_reg: float = 10.0,
        gender_reg: float = 25.0,
        mf_reg: float = 20.0,
        n_epochs: int = 10,
        seed: int = 0,
    ) -> None:
        self.n_factors = n_factors
        self.bias_reg = bias_reg
        self.gender_reg = gender_reg
        self.mf_reg = mf_reg
        self.n_epochs = n_epochs
        self.seed = seed

    def fit(self, data: Dataset) -> "RatingModel":
        u = data.ratings["user"].to_numpy()
        i = data.ratings["item"].to_numpy()
        r = data.ratings["rating"].to_numpy(dtype=float)
        n_users, n_items = data.n_users, data.n_items
        self._gender = data.users["gender"].to_numpy()  # per-user "M"/"F"

        # --- stage 1: biases (alternating shrunk means) ---
        self.mu = r.mean()
        b_u = np.zeros(n_users)
        b_i = np.zeros(n_items)
        cnt_u = np.bincount(u, minlength=n_users)
        cnt_i = np.bincount(i, minlength=n_items)
        for _ in range(self.n_epochs):
            b_i = np.bincount(i, weights=r - self.mu - b_u[u], minlength=n_items)
            b_i /= self.bias_reg + cnt_i
            b_u = np.bincount(u, weights=r - self.mu - b_i[i], minlength=n_users)
            b_u /= self.bias_reg + cnt_u
        self.b_u, self.b_i = b_u, b_i

        # --- stage 2: gender-item effect on residuals ---
        resid = r - self.mu - b_u[u] - b_i[i]
        self.delta = np.zeros((len(GENDERS), n_items))
        for g_idx, g in enumerate(GENDERS):
            mask = self._gender[u] == g
            s = np.bincount(i[mask], weights=resid[mask], minlength=n_items)
            c = np.bincount(i[mask], minlength=n_items)
            self.delta[g_idx] = s / (self.gender_reg + c)
        g_of_u = np.array([GENDERS.index(g) for g in self._gender])
        resid -= self.delta[g_of_u[u], i]

        # --- stage 3: ALS matrix factorization on the remainder ---
        rng = np.random.default_rng(self.seed)
        P = rng.normal(scale=0.05, size=(n_users, self.n_factors))
        Q = rng.normal(scale=0.05, size=(n_items, self.n_factors))
        by_user = pd.DataFrame({"u": u, "i": i, "e": resid}).groupby("u")
        by_item = pd.DataFrame({"u": u, "i": i, "e": resid}).groupby("i")
        user_groups = {k: (v["i"].to_numpy(), v["e"].to_numpy()) for k, v in by_user}
        item_groups = {k: (v["u"].to_numpy(), v["e"].to_numpy()) for k, v in by_item}
        eye = self.mf_reg * np.eye(self.n_factors)
        for _ in range(self.n_epochs):
            for uu, (ii, ee) in user_groups.items():
                Qi = Q[ii]
                P[uu] = np.linalg.solve(Qi.T @ Qi + eye, Qi.T @ ee)
            for ii, (uu, ee) in item_groups.items():
                Pu = P[uu]
                Q[ii] = np.linalg.solve(Pu.T @ Pu + eye, Pu.T @ ee)
        self.P, self.Q = P, Q

        pred = self._predict_pairs(u, i, g_of_u)
        self.train_rmse = float(np.sqrt(np.mean((r - pred) ** 2)))
        return self

    def _predict_pairs(self, u: np.ndarray, i: np.ndarray, g_of_u: np.ndarray) -> np.ndarray:
        return (
            self.mu + self.b_u[u] + self.b_i[i]
            + self.delta[g_of_u[u], i]
            + np.einsum("uf,uf->u", self.P[u], self.Q[i])
        )

    def predict_matrix(self, gender_overrides: dict[int, str] | None = None) -> np.ndarray:
        """Full (n_users, n_items) prediction matrix, clipped to [1, 5].

        ``gender_overrides`` maps 0-based user index to "M"/"F" and only
        swaps the demographic pathway for those users.
        """
        gender = self._gender.copy()
        if gender_overrides:
            for uu, g in gender_overrides.items():
                if g not in GENDERS:
                    raise ValueError(f"gender must be one of {GENDERS}, got {g!r}")
                gender[uu] = g
        g_of_u = np.array([GENDERS.index(g) for g in gender])
        r_hat = (
            self.mu
            + self.b_u[:, None]
            + self.b_i[None, :]
            + self.delta[g_of_u]
            + self.P @ self.Q.T
        )
        return np.clip(r_hat, 1.0, 5.0)
