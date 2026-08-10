"""Comparison of a baseline and a modified solution.

Produces the quantitative report that grounds the LLM explanation: the
explainer is only allowed to talk about numbers that appear here, which
keeps explanations faithful to the solver output.
"""

from __future__ import annotations

import numpy as np

from .data import Dataset
from .problem import Solution


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array (0 = equal, 1 = concentrated)."""
    total = float(values.sum())
    if total == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cum = np.cumsum(sorted_vals)
    return float((n + 1 - 2 * cum.sum() / total) / n)


def _dist(values: np.ndarray, gini: bool = False) -> dict:
    """Summary statistics of a per-user (or per-item) array."""
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    out = {
        "min": round(float(values.min()), 3),
        "p25": round(float(q25), 3),
        "median": round(float(q50), 3),
        "p75": round(float(q75), 3),
        "max": round(float(values.max()), 3),
        "std": round(float(values.std()), 3),
    }
    if gini:
        out["gini"] = round(_gini(values), 3)
    return out


def compare_solutions(
    base: Solution,
    mod: Solution,
    data: Dataset,
    focal_users: list[int] | None = None,
) -> dict:
    changed = np.array([
        not np.array_equal(np.sort(base.recs[u]), np.sort(mod.recs[u]))
        for u in range(data.n_users)
    ])
    cold = data.cold_items
    report = {
        "objective": {
            "base": round(base.objective, 1),
            "modified": round(mod.objective, 1),
            "delta_pct": round(100 * (mod.objective - base.objective) / base.objective, 2),
        },
        "mean_predicted_rating": {
            "base": round(base.mean_predicted_rating, 3),
            "modified": round(mod.mean_predicted_rating, 3),
        },
        "users_with_changed_slate": {
            "count": int(changed.sum()),
            "fraction": round(float(changed.mean()), 3),
        },
        "cold_item_exposure": {
            "base_total": round(float(base.exposure[cold].sum()), 1),
            "modified_total": round(float(mod.exposure[cold].sum()), 1),
            "base_items_shown": int((base.exposure[cold] > 1e-6).sum()),
            "modified_items_shown": int((mod.exposure[cold] > 1e-6).sum()),
            "n_cold_items": len(cold),
            "mean_cold_items_per_user_slate": {
                "base": round(float(base.exposure[cold].sum()) / data.n_users, 2),
                "modified": round(float(mod.exposure[cold].sum()) / data.n_users, 2),
            },
        },
        "per_user_slate_rating_distribution": {
            "note": "mean predicted rating of each user's slate; min = worst-off user",
            "base": _dist(base.user_mean_rating),
            "modified": _dist(mod.user_mean_rating),
        },
        "exploration_burden_distribution": {
            "note": "number of cold items in each user's slate; gini 0 = evenly shared",
            "base": _dist(base.user_cold_items, gini=True),
            "modified": _dist(mod.user_cold_items, gini=True),
        },
        "item_exposure_concentration": {
            "note": "how unevenly total exposure is spread over the catalog",
            "gini": {
                "base": round(_gini(base.exposure), 3),
                "modified": round(_gini(mod.exposure), 3),
            },
            "top_10pct_items_exposure_share": {
                "base": _top_share(base.exposure),
                "modified": _top_share(mod.exposure),
            },
        },
        "fractional_entries": {"base": base.n_fractional, "modified": mod.n_fractional},
        "slate_size": {"base": base.slate_size, "modified": mod.slate_size},
    }
    if focal_users:
        report["focal_users"] = {
            str(u): _focal_diff(base, mod, data, u) for u in focal_users
        }
    return report


def _top_share(exposure: np.ndarray) -> float:
    """Share of total exposure captured by the 10% most-exposed items."""
    top_n = max(1, len(exposure) // 10)
    top = np.sort(exposure)[-top_n:]
    return round(float(top.sum() / exposure.sum()), 3)


def _focal_diff(base: Solution, mod: Solution, data: Dataset, u: int) -> dict:
    cold = set(data.cold_items.tolist())
    before = set(base.recs[u].tolist())
    after = set(mod.recs[u].tolist())

    def label(i: int) -> str:
        return data.title(i) + (" [cold]" if i in cold else "")

    user = data.users.loc[u]
    own_ratings = data.ratings[data.ratings["user"] == u]["rating"]
    return {
        "profile": {
            "note": ("attributes as recorded in the dataset; counterfactual "
                     "overrides, if any, are described in the applied change"),
            "age": int(user["age"]),
            "gender": str(user["gender"]),
            "occupation": str(user["occupation"]),
            "n_ratings": int(len(own_ratings)),
            "mean_rating_given": round(float(own_ratings.mean()), 2),
        },
        "cold_items_in_slate": {
            "base": len(before & cold),
            "modified": len(after & cold),
        },
        "mean_predicted_slate_rating": {
            "base": round(float(base.user_mean_rating[u]), 3),
            "modified": round(float(mod.user_mean_rating[u]), 3),
        },
        "kept": sorted(label(i) for i in before & after),
        "removed": sorted(label(i) for i in before - after),
        "added": sorted(label(i) for i in after - before),
    }


def report_text(report: dict) -> str:
    """Render the report as compact indented text for the LLM prompt."""
    lines: list[str] = []

    def walk(node: dict, indent: int) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                lines.append("  " * indent + f"{key}:")
                walk(value, indent + 1)
            else:
                lines.append("  " * indent + f"{key}: {value}")

    walk(report, 0)
    return "\n".join(lines)
