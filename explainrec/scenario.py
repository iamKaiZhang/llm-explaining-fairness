"""Scenarios and the modification schema the LLM emits.

A ``Scenario`` is one fully-specified optimization problem: the rating
model inputs (including any attribute counterfactuals), the active
constraint specs, and the slate size. A ``Modification`` is the
structured output of the LLM interpreter and maps one scenario to
another; applying it never mutates the original, so baseline and
counterfactual can always be compared side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from pydantic import BaseModel, Field

from .constraints import ConstraintSpec
from .data import Dataset
from .problem import Solution, solve_allocation
from .ratings import RatingModel


class GenderOverride(BaseModel):
    user_id: int = Field(description="0-based user index")
    gender: str = Field(description='counterfactual gender, "M" or "F"')


class Modification(BaseModel):
    """Structured problem edit produced by the LLM interpreter."""

    summary: str = Field(
        description="one sentence restating the change in plain language"
    )
    add_constraints: list[ConstraintSpec] = Field(default_factory=list)
    remove_constraints: list[str] = Field(
        default_factory=list, description="names of active constraints to drop"
    )
    gender_overrides: list[GenderOverride] = Field(default_factory=list)
    set_slate_size: int | None = Field(
        default=None, description="only if the query asks to change the slate size"
    )
    focal_users: list[int] = Field(
        default_factory=list,
        description="0-based user indices the query is about, if it concerns specific users",
    )

    def is_noop(self) -> bool:
        return not (
            self.add_constraints
            or self.remove_constraints
            or self.gender_overrides
            or self.set_slate_size
        )


@dataclass
class Scenario:
    model: RatingModel
    data: Dataset
    constraints: list[ConstraintSpec]
    slate_size: int = 10
    gender_overrides: dict[int, str] = field(default_factory=dict)

    def apply(self, mod: Modification) -> "Scenario":
        names = {c.name for c in self.constraints}
        unknown = [n for n in mod.remove_constraints if n not in names]
        if unknown:
            raise ValueError(f"cannot remove unknown constraints: {unknown}")
        new_constraints = [
            c for c in self.constraints if c.name not in mod.remove_constraints
        ]
        for spec in mod.add_constraints:
            if spec.name in {c.name for c in new_constraints}:
                raise ValueError(f"constraint name {spec.name!r} already in use")
            new_constraints.append(spec)
        overrides = dict(self.gender_overrides)
        for o in mod.gender_overrides:
            if not 0 <= o.user_id < self.data.n_users:
                raise ValueError(
                    f"gender override: user_id {o.user_id} out of range "
                    f"(0..{self.data.n_users - 1})"
                )
            if o.gender not in ("M", "F"):
                raise ValueError(f'gender must be "M" or "F", got {o.gender!r}')
            overrides[o.user_id] = o.gender
        return replace(
            self,
            constraints=new_constraints,
            slate_size=mod.set_slate_size or self.slate_size,
            gender_overrides=overrides,
        )

    def solve(self) -> Solution:
        r_hat = self.model.predict_matrix(self.gender_overrides or None)
        return solve_allocation(r_hat, self.constraints, self.data, self.slate_size)

    def describe_constraints(self) -> str:
        if not self.constraints:
            return "(none)"
        return "\n".join(c.describe(self.data) for c in self.constraints)
