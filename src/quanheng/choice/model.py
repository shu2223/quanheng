from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from quanheng._base import Model
from quanheng.quantity import Quantity
from quanheng.weights import WeightSpec

Id = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]+$")]


class Option(Model):
    id: Id
    name: str
    note: str = ""


class Criterion(Model):
    id: Id
    name: str
    unit: str = ""
    bad: float
    good: float
    cap_at_good: bool = False


class Constraint(Model):
    criterion: str
    op: Literal["<=", ">="]
    threshold: float


class ChoiceModel(Model):
    kind: Literal["choice"] = "choice"
    schema_version: Literal[1] = 1
    title: str
    description: str = ""
    options: list[Option]
    criteria: list[Criterion]
    constraints: list[Constraint] = Field(default_factory=list)
    scores: dict[str, dict[str, Quantity]]  # option id -> criterion id -> value
    weights: WeightSpec

    @model_validator(mode="after")
    def _check_structure(self) -> "ChoiceModel":
        if len(self.options) < 2:
            raise ValueError("options 至少需要 2 个候选")
        if not self.criteria:
            raise ValueError("criteria 至少需要 1 条准则")
        option_ids = [o.id for o in self.options]
        criterion_ids = [c.id for c in self.criteria]
        for ids, place in ((option_ids, "options"), (criterion_ids, "criteria")):
            for i, x in enumerate(ids):
                if x in ids[:i]:
                    raise ValueError(f"{place} 中 id {x} 重复")
        for i, c in enumerate(self.criteria):
            if c.bad == c.good:
                raise ValueError(f"criteria.{i} 的 bad 和 good 不能相等")
        for oid in option_ids:
            if oid not in self.scores:
                raise ValueError(f"scores 缺少候选 {oid}")
        for oid in self.scores:
            if oid not in option_ids:
                raise ValueError(f"scores 多出候选 {oid}")
        for oid in option_ids:
            row = self.scores[oid]
            for cid in criterion_ids:
                if cid not in row:
                    raise ValueError(f"scores.{oid} 缺少准则 {cid}")
            for cid in row:
                if cid not in criterion_ids:
                    raise ValueError(f"scores.{oid} 多出准则 {cid}")
        for i, k in enumerate(self.constraints):
            if k.criterion not in criterion_ids:
                raise ValueError(f"constraints.{i} 引用了不存在的准则 {k.criterion}")
        for cid in criterion_ids:
            if cid not in self.weights.values:
                raise ValueError(f"weights.values 缺少准则 {cid}")
        for cid in self.weights.values:
            if cid not in criterion_ids:
                raise ValueError(f"weights.values 多出准则 {cid}")
        return self

    def unconfirmed_inputs(self) -> list[str]:
        return [
            f"scores.{o.id}.{c.id}"
            for o in self.options
            for c in self.criteria
            if self.scores[o.id][c.id].needs_confirmation
        ]
