"""Constraint specifications and their compilation to cvxpy constraints.

A ``ConstraintSpec`` is a declarative, JSON-serializable description of
one constraint on the allocation variable X (n_users x n_items). The
LLM emits these specs; ``build_constraint`` compiles them against a
concrete cvxpy variable. Adding a new constraint type means adding a
``type`` literal, its parameters, and a branch in ``build_constraint``.
"""

from __future__ import annotations

from typing import Literal

import cvxpy as cp
import numpy as np
from pydantic import BaseModel, Field, model_validator

from .data import Dataset


class ItemSelector(BaseModel):
    """Selects a set of items. Exactly the fields for ``kind`` are used."""

    kind: Literal["cold", "ids", "genre", "popular", "all"]
    ids: list[int] | None = Field(
        default=None, description="0-based item indices, for kind='ids'"
    )
    genre: str | None = Field(
        default=None, description="genre name, for kind='genre'"
    )
    top: int | None = Field(
        default=None, description="number of most-rated items, for kind='popular'"
    )

    def resolve(self, data: Dataset) -> np.ndarray:
        if self.kind == "cold":
            return data.cold_items
        if self.kind == "ids":
            if not self.ids:
                raise ValueError("kind='ids' requires a non-empty ids list")
            bad = [i for i in self.ids if not 0 <= i < data.n_items]
            if bad:
                raise ValueError(
                    f"item ids out of range (0..{data.n_items - 1}): {bad}"
                )
            return np.asarray(self.ids, dtype=int)
        if self.kind == "genre":
            if not self.genre:
                raise ValueError("kind='genre' requires a genre name")
            return data.items_with_genre(self.genre)
        if self.kind == "popular":
            if not self.top:
                raise ValueError("kind='popular' requires top")
            return data.popular_items(self.top)
        return np.arange(data.n_items)


class ConstraintSpec(BaseModel):
    """One declarative constraint on the allocation.

    Types:
    - min_item_exposure: each selected item is recommended to at least
      ``min_users`` users (sum over users of x[u,i] >= min_users).
    - max_item_exposure: each selected item is recommended to at most
      ``max_users`` users.
    - forbid_items: selected items are never recommended (to ``user_id``
      if given, otherwise to anyone).
    - force_assign: item ``item_id`` must be recommended to user ``user_id``.
    """

    name: str = Field(description="unique handle, used to remove the constraint later")
    type: Literal[
        "min_item_exposure", "max_item_exposure", "forbid_items", "force_assign"
    ]
    items: ItemSelector | None = None
    min_users: int | None = None
    max_users: int | None = None
    user_id: int | None = Field(default=None, description="0-based user index")
    item_id: int | None = Field(default=None, description="0-based item index")

    @model_validator(mode="after")
    def _check_params(self) -> "ConstraintSpec":
        needs = {
            "min_item_exposure": ["items", "min_users"],
            "max_item_exposure": ["items", "max_users"],
            "forbid_items": ["items"],
            "force_assign": ["user_id", "item_id"],
        }[self.type]
        missing = [f for f in needs if getattr(self, f) is None]
        if missing:
            raise ValueError(f"constraint {self.name!r} ({self.type}) missing {missing}")
        return self

    def describe(self, data: Dataset) -> str:
        if self.type == "min_item_exposure":
            n = len(self.items.resolve(data))
            return (
                f"[{self.name}] each of {n} items ({self.items.kind}) reaches "
                f"at least {self.min_users} users"
            )
        if self.type == "max_item_exposure":
            n = len(self.items.resolve(data))
            return (
                f"[{self.name}] each of {n} items ({self.items.kind}) reaches "
                f"at most {self.max_users} users"
            )
        if self.type == "forbid_items":
            n = len(self.items.resolve(data))
            who = f"user {self.user_id}" if self.user_id is not None else "everyone"
            return f"[{self.name}] {n} items ({self.items.kind}) hidden from {who}"
        return f"[{self.name}] user {self.user_id} must receive item {self.item_id}"


def build_constraint(
    spec: ConstraintSpec, X: cp.Variable, data: Dataset, slate_size: int
) -> list[cp.Constraint]:
    if spec.type == "min_item_exposure":
        items = spec.items.resolve(data)
        # feasibility guard: total forced slots cannot exceed total slots
        if spec.min_users * len(items) > slate_size * data.n_users:
            raise ValueError(
                f"constraint {spec.name!r} infeasible: {len(items)} items x "
                f"{spec.min_users} users > {slate_size * data.n_users} total slots"
            )
        return [cp.sum(X[:, items], axis=0) >= spec.min_users]
    if spec.type == "max_item_exposure":
        items = spec.items.resolve(data)
        return [cp.sum(X[:, items], axis=0) <= spec.max_users]
    if spec.type == "forbid_items":
        items = spec.items.resolve(data)
        if spec.user_id is not None:
            _check_user(spec.user_id, data, spec.name)
            return [X[spec.user_id, items] == 0]
        return [X[:, items] == 0]
    if spec.type == "force_assign":
        _check_user(spec.user_id, data, spec.name)
        if not 0 <= spec.item_id < data.n_items:
            raise ValueError(
                f"constraint {spec.name!r}: item_id {spec.item_id} out of "
                f"range (0..{data.n_items - 1})"
            )
        return [X[spec.user_id, spec.item_id] == 1]
    raise ValueError(f"unknown constraint type {spec.type!r}")


def _check_user(user_id: int, data: Dataset, name: str) -> None:
    if not 0 <= user_id < data.n_users:
        raise ValueError(
            f"constraint {name!r}: user_id {user_id} out of range "
            f"(0..{data.n_users - 1})"
        )
