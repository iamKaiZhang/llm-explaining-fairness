"""Comparison of a baseline and a modified solution.

Produces the quantitative report that grounds the LLM explanation: the
explainer is only allowed to talk about numbers that appear here, which
keeps explanations faithful to the solver output.
"""

from __future__ import annotations

import numpy as np

from .data import Dataset
from .problem import Solution


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
        },
        "fractional_entries": {"base": base.n_fractional, "modified": mod.n_fractional},
        "slate_size": {"base": base.slate_size, "modified": mod.slate_size},
    }
    if focal_users:
        report["focal_users"] = {
            str(u): _focal_diff(base, mod, data, u) for u in focal_users
        }
    return report


def _focal_diff(base: Solution, mod: Solution, data: Dataset, u: int) -> dict:
    before = set(base.recs[u].tolist())
    after = set(mod.recs[u].tolist())
    return {
        "kept": sorted(data.title(i) for i in before & after),
        "removed": sorted(data.title(i) for i in before - after),
        "added": sorted(data.title(i) for i in after - before),
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
