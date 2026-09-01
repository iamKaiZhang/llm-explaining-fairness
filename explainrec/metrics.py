"""Measurement metrics over a solved scenario and the rating model.

These functions are the *executable ground-truth definitions* for the
measurement-query dataset in ``datasets/``: each question set references
one function here, and the gold answers are computed by calling it, never
typed by hand. When the eventual measurement tool answers a query, it must
call the same function with the same parameters.

Semantics (documented once, referenced by the dataset):

- An item is "recommended" if its exposure in the solution is >= 1 user
  (solutions are integral in practice; the 1e-6 slack covers numerics).
- Item similarity is genre Jaccard: |G_i & G_j| / |G_i | G_j| over the
  19 MovieLens genre flags. Deterministic and model-free.
- The exposure gap of a pair is relative: |e_i - e_j| / max(e_i, e_j).
- Per-user accuracy metrics are IN-SAMPLE (computed on the user's observed
  ratings with the fitted model's clipped predictions); there is no
  train/test split yet. Ranking metrics rank each user's observed items by
  predicted rating with relevance = (observed rating >= 4), cut at k.
  Users with no relevant item are excluded (NaN) from Recall and nDCG.
- Generalized cross-entropy (GCE, Deldjoo et al. 2019) compares the
  normalized per-user benefit distribution to the uniform one:
  GCE = (sum_u f_u^beta p_u^(1-beta) - 1) / (beta (1 - beta)), f_u = 1/n.
  GCE <= 0, and 0 means perfectly equal. Default beta = 0.5, which stays
  finite when some users have zero benefit (beta >= 1 would not).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Dataset, GENRES
from .problem import Solution
from .ratings import RatingModel

EXPOSED_EPS = 1e-6

USER_METRICS = ["MAE", "RMSE", "nDCG", "Precision", "Recall"]


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------

def coverage(solution: Solution, data: Dataset) -> dict:
    """Fraction of catalog items recommended to at least one user."""
    n_recommended = int((solution.exposure >= 1 - EXPOSED_EPS).sum())
    return {
        "fraction": n_recommended / data.n_items,
        "n_recommended": n_recommended,
        "n_items": data.n_items,
    }


# --------------------------------------------------------------------------
# coverage disparity (individual fairness over items)
# --------------------------------------------------------------------------

def genre_jaccard(data: Dataset) -> np.ndarray:
    """(n_items, n_items) Jaccard similarity of genre sets."""
    flags = np.zeros((data.n_items, len(GENRES)), dtype=float)
    for i, genres in enumerate(data.items["genres"]):
        for g in genres:
            flags[i, GENRES.index(g)] = 1.0
    inter = flags @ flags.T
    sizes = flags.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def coverage_disparity(
    solution: Solution, data: Dataset, alpha: float, epsilon: float,
) -> dict:
    """Fraction of alpha-similar recommended item pairs whose relative
    exposure gap is at most epsilon.

    Individual fairness reading: any two alpha-similar recommended items
    should receive similar coverage; this measures how often that holds.
    """
    exposed = np.flatnonzero(solution.exposure >= 1 - EXPOSED_EPS)
    sim = genre_jaccard(data)[np.ix_(exposed, exposed)]
    e = solution.exposure[exposed]
    gap = np.abs(e[:, None] - e[None, :]) / np.maximum(e[:, None], e[None, :])

    iu = np.triu_indices(len(exposed), k=1)
    similar = sim[iu] >= alpha - 1e-12
    n_similar = int(similar.sum())
    if n_similar == 0:
        return {"fraction": 1.0, "n_similar_pairs": 0, "n_within_gap": 0,
                "n_recommended": len(exposed)}
    within = gap[iu][similar] <= epsilon + 1e-12
    return {
        "fraction": float(within.mean()),
        "n_similar_pairs": n_similar,
        "n_within_gap": int(within.sum()),
        "n_recommended": len(exposed),
    }


# --------------------------------------------------------------------------
# per-user accuracy metrics (in-sample) and inequality
# --------------------------------------------------------------------------

@dataclass
class UserMetrics:
    """Per-user arrays (NaN where a metric is undefined for a user)."""
    values: dict[str, np.ndarray]
    k: int
    relevance_threshold: float


def per_user_metrics(
    model: RatingModel, data: Dataset, k: int = 10, relevance_threshold: float = 4.0,
) -> UserMetrics:
    r_hat = model.predict_matrix()
    u = data.ratings["user"].to_numpy()
    i = data.ratings["item"].to_numpy()
    r = data.ratings["rating"].to_numpy(dtype=float)
    pred = r_hat[u, i]

    out = {m: np.full(data.n_users, np.nan) for m in USER_METRICS}
    order = np.argsort(u, kind="stable")
    u_s, i_s, r_s, p_s = u[order], i[order], r[order], pred[order]
    bounds = np.searchsorted(u_s, np.arange(data.n_users + 1))

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for user in range(data.n_users):
        lo, hi = bounds[user], bounds[user + 1]
        if lo == hi:
            continue
        err = r_s[lo:hi] - p_s[lo:hi]
        out["MAE"][user] = np.abs(err).mean()
        out["RMSE"][user] = np.sqrt((err ** 2).mean())

        rel = (r_s[lo:hi] >= relevance_threshold).astype(float)
        rank = np.argsort(-p_s[lo:hi], kind="stable")
        top = rel[rank][:k]
        out["Precision"][user] = top.sum() / k
        n_rel = rel.sum()
        if n_rel > 0:
            out["Recall"][user] = top.sum() / n_rel
            dcg = float((top * discounts[: len(top)]).sum())
            ideal = np.sort(rel)[::-1][:k]
            idcg = float((ideal * discounts[: len(ideal)]).sum())
            out["nDCG"][user] = dcg / idcg
    return UserMetrics(values=out, k=k, relevance_threshold=relevance_threshold)


def gce(values: np.ndarray, beta: float = 0.5) -> dict:
    """Generalized cross-entropy of a per-user benefit array against the
    uniform (fair) distribution. 0 = perfectly equal; more negative = more
    unequal. NaN entries (users for whom the metric is undefined) are
    excluded. ``beta`` must not be 0 or 1; the 0.5 default stays finite
    when some benefits are zero.
    """
    if beta in (0.0, 1.0):
        raise ValueError("beta must not be 0 or 1")
    v = values[~np.isnan(values)]
    total = float(v.sum())
    n = len(v)
    if n == 0 or total == 0:
        return {"gce": 0.0, "n_users": n}
    p = v / total
    f = 1.0 / n
    s = float(np.sum(f ** beta * p ** (1.0 - beta)))
    return {"gce": (s - 1.0) / (beta * (1.0 - beta)), "n_users": n}


def benefit_inequality(
    model: RatingModel, data: Dataset, metric: str,
    beta: float = 0.5, k: int = 10,
) -> dict:
    """GCE inequality across users with respect to one accuracy metric."""
    if metric not in USER_METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {USER_METRICS}")
    per_user = per_user_metrics(model, data, k=k)
    values = per_user.values[metric]
    result = gce(values, beta=beta)
    result.update({
        "metric": metric, "beta": beta, "k": k,
        "mean": round(float(np.nanmean(values)), 4),
    })
    return result


# --------------------------------------------------------------------------
# item and user groups
# --------------------------------------------------------------------------

_YEAR_RE = __import__("re").compile(r"\((\d{4})\)")


def _title_year(title: str) -> int | None:
    m = _YEAR_RE.search(title)
    return int(m.group(1)) if m else None


def item_group(data: Dataset, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Resolve an item-group spec to (group indices, complement indices).

    Specs: {"kind": "genre", "genre": <name>} — complement = all other items;
           {"kind": "era", "before": <year>} — complement = year >= before
             (items without a parseable year excluded from both sides);
           {"kind": "cold"} — complement = non-cold items.
    """
    if spec["kind"] == "genre":
        group = data.items_with_genre(spec["genre"])
        mask = np.zeros(data.n_items, dtype=bool)
        mask[group] = True
        return group, np.flatnonzero(~mask)
    if spec["kind"] == "era":
        years = np.array([
            _title_year(t) or -1 for t in data.items["title"]
        ])
        return (np.flatnonzero((years >= 0) & (years < spec["before"])),
                np.flatnonzero(years >= spec["before"]))
    if spec["kind"] == "cold":
        mask = np.zeros(data.n_items, dtype=bool)
        mask[data.cold_items] = True
        return data.cold_items, np.flatnonzero(~mask)
    raise ValueError(f"unknown item group kind {spec['kind']!r}")


def user_group(data: Dataset, spec: dict) -> np.ndarray:
    """Resolve a user-group spec to user indices.

    Specs: {"attribute": "gender", "value": "M"|"F"};
           {"attribute": "occupation", "value": <name>};
           {"attribute": "age", "min": a, "max": b} (inclusive bounds).
    """
    users = data.users
    if spec["attribute"] == "gender":
        mask = users["gender"] == spec["value"]
    elif spec["attribute"] == "occupation":
        mask = users["occupation"] == spec["value"]
    elif spec["attribute"] == "age":
        mask = ((users["age"] >= spec.get("min", 0))
                & (users["age"] <= spec.get("max", 200)))
    else:
        raise ValueError(f"unknown user attribute {spec['attribute']!r}")
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) == 0:
        raise ValueError(f"user group {spec} matches no users")
    return idx


# --------------------------------------------------------------------------
# provider-side group fairness: within-group coverage gap
# --------------------------------------------------------------------------

def group_coverage_gap(solution: Solution, data: Dataset, spec: dict) -> dict:
    """|coverage within group - coverage within complement|, where coverage
    within a group is the fraction of that group's items recommended to at
    least one user."""
    group, complement = item_group(data, spec)
    exposed = solution.exposure >= 1 - EXPOSED_EPS

    def frac(idx: np.ndarray) -> float:
        return float(exposed[idx].mean()) if len(idx) else 0.0

    g, c = frac(group), frac(complement)
    return {
        "gap": abs(g - c),
        "group_fraction": round(g, 4), "complement_fraction": round(c, 4),
        "n_group": len(group), "n_complement": len(complement),
    }


# --------------------------------------------------------------------------
# user-side group fairness: demographic parity and worst group
# --------------------------------------------------------------------------

def _top_k_slate(solution: Solution, r_hat: np.ndarray, u: int, k: int) -> np.ndarray:
    """The user's k slate items with the highest predicted rating."""
    slate = solution.recs[u]
    order = np.argsort(-r_hat[u, slate], kind="stable")
    return slate[order[:k]]


def _liked_proportion(
    solution: Solution, r_hat: np.ndarray, users: np.ndarray, k: int, stars: float,
) -> float:
    """Mean over users of the fraction of their top-k slate items with
    predicted rating >= stars ("would rate at least `stars`" is defined on
    the model's predictions; recommended items have no observed ratings)."""
    props = [
        float((r_hat[u, _top_k_slate(solution, r_hat, u, k)] >= stars).mean())
        for u in users
    ]
    return float(np.mean(props))


def demographic_parity(
    solution: Solution, r_hat: np.ndarray, data: Dataset,
    group_x: dict, group_y: dict, k: int, stars: float, bound: float,
) -> dict:
    """Gap between two user groups in the mean proportion of their top-k
    recommendations predicted at >= stars; parity holds if gap <= bound."""
    px = _liked_proportion(solution, r_hat, user_group(data, group_x), k, stars)
    py = _liked_proportion(solution, r_hat, user_group(data, group_y), k, stars)
    return {
        "gap": abs(px - py),
        "proportion_x": round(px, 4), "proportion_y": round(py, 4),
        "parity_holds": bool(abs(px - py) <= bound + 1e-12),
        "k": k, "stars": stars, "bound": bound,
    }


UTILITIES = ["mean_predicted_rating", "fraction_predicted_at_least_4"]


def worst_group(
    solution: Solution, r_hat: np.ndarray, data: Dataset,
    groups: list[tuple[str, dict]], k: int, utility: str, bound: float,
) -> dict:
    """Per-group utility of the top-k recommendations; identifies the
    worst-served and best-served groups and whether the worst is within
    `bound` of the best (maximin fairness reading)."""
    if utility not in UTILITIES:
        raise ValueError(f"unknown utility {utility!r}; expected one of {UTILITIES}")
    per_group: dict[str, float] = {}
    for label, spec in groups:
        users = user_group(data, spec)
        if utility == "mean_predicted_rating":
            vals = [float(r_hat[u, _top_k_slate(solution, r_hat, u, k)].mean())
                    for u in users]
            per_group[label] = float(np.mean(vals))
        else:
            per_group[label] = _liked_proportion(solution, r_hat, users, k, 4.0)
    worst = min(per_group, key=per_group.get)
    best = max(per_group, key=per_group.get)
    gap = per_group[best] - per_group[worst]
    return {
        "worst_group": worst, "best_group": best,
        "gap": gap,
        "within_bound": bool(gap <= bound + 1e-12),
        "per_group": {g: round(v, 4) for g, v in per_group.items()},
        "k": k, "utility": utility, "bound": bound,
    }
