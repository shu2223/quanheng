import copy
from typing import Any, Literal

from pydantic import ValidationError

from quanheng import zh
from quanheng._base import Model
from quanheng.choice.model import ChoiceModel
from quanheng.choice.result import ChoiceResult, Thresholds
from quanheng.choice.solve import solve_choice
from quanheng.errors import PatchError


class PatchOp(Model):
    target: str
    op: Literal["set", "scale", "shift"]
    value: float


class Change(Model):
    target: str
    before: list[float]
    after: list[float]


class WhatIfResult(Model):
    changes: list[Change]
    before: ChoiceResult
    after: ChoiceResult
    win_rate_delta: dict[str, float]
    winner_changed: bool
    level_changed: bool


def _arith(op: PatchOp, x: float) -> float:
    if op.op == "scale":
        return x * op.value
    if op.op == "shift":
        return x + op.value
    return op.value


def _numbers(value: dict[str, Any]) -> list[float]:
    kind = value["kind"]
    if kind == "exact":
        return [float(value["value"])]
    if kind == "range":
        return [float(value["low"]), float(value["high"])]
    return [float(value["low"]), float(value["mode"]), float(value["high"])]


def _check_factor(op: PatchOp) -> None:
    if op.op == "scale" and not op.value > 0:
        raise PatchError(f"{op.target}: scale 的系数必须大于 0")


def _patch_quantity(q: dict[str, Any], op: PatchOp) -> Change:
    _check_factor(op)
    before = _numbers(q["value"])
    if op.op == "set":
        q["value"] = {"kind": "exact", "value": op.value}
        q["source"] = {"type": "user_input"}
    else:
        for key in ("value", "low", "mode", "high"):
            if key in q["value"]:
                q["value"][key] = _arith(op, q["value"][key])
    return Change(target=op.target, before=before, after=_numbers(q["value"]))


def _patch_weight(data: dict[str, Any], criterion: str, op: PatchOp) -> Change:
    _check_factor(op)
    values: dict[str, float] = data["weights"]["values"]
    total = sum(values.values())
    norm = {k: x / total for k, x in values.items()}
    old = norm[criterion]
    if len(norm) == 1 or old == 1:
        raise PatchError(f"{op.target}: 只有这一条权重非零，无法调整")
    v = _arith(op, old)
    if not 0 <= v <= 1:
        raise PatchError(f"{op.target}: 调整后的权重 {v} 不在 0 到 1 之间")
    k = (1 - v) / (1 - old)
    data["weights"]["values"] = {c: (v if c == criterion else x * k) for c, x in norm.items()}
    return Change(target=op.target, before=[old], after=[v])


def _apply_one(data: dict[str, Any], op: PatchOp) -> Change:
    parts = op.target.split(".")
    head = parts[0]
    if head == "criteria":
        raise PatchError(zh.PATCH_ANCHOR)
    if head == "scores" and len(parts) == 3:
        row = data["scores"].get(parts[1])
        if row is None or parts[2] not in row:
            raise PatchError(f"目标不存在: {op.target}")
        return _patch_quantity(row[parts[2]], op)
    if head == "weights" and len(parts) == 2:
        if parts[1] not in data["weights"]["values"]:
            raise PatchError(f"目标不存在: {op.target}")
        return _patch_weight(data, parts[1], op)
    if head == "constraints" and len(parts) == 3 and parts[2] == "threshold":
        constraints = data["constraints"]
        index = parts[1]
        # isdigit() alone accepts characters such as "²" that int() rejects.
        if not (index.isascii() and index.isdigit()) or int(index) >= len(constraints):
            raise PatchError(f"目标不存在: {op.target}")
        _check_factor(op)
        c = constraints[int(index)]
        before = float(c["threshold"])
        c["threshold"] = _arith(op, before)
        return Change(target=op.target, before=[before], after=[c["threshold"]])
    raise PatchError(f"不支持的目标: {op.target}")


def apply_patch(model: ChoiceModel, ops: list[PatchOp]) -> tuple[ChoiceModel, list[Change]]:
    data = copy.deepcopy(model.model_dump())
    try:
        changes = [_apply_one(data, op) for op in ops]
        return ChoiceModel.model_validate(data), changes
    except ValidationError as exc:
        raise PatchError(f"修改后的模型无效: {exc}") from exc


def compare(
    model: ChoiceModel,
    ops: list[PatchOp],
    *,
    samples: int = 10_000,
    seed: int = 0,
    thresholds: Thresholds | None = None,
) -> WhatIfResult:
    patched, changes = apply_patch(model, ops)
    before = solve_choice(model, samples=samples, seed=seed, thresholds=thresholds)
    after = solve_choice(patched, samples=samples, seed=seed, thresholds=thresholds)
    options = set(before.win_rates) | set(after.win_rates)
    delta = {
        o: float(after.win_rates.get(o, 0.0)) - float(before.win_rates.get(o, 0.0))
        for o in sorted(options)
    }
    return WhatIfResult(
        changes=changes,
        before=before,
        after=after,
        win_rate_delta=delta,
        winner_changed=before.verdict.winner != after.verdict.winner,
        level_changed=before.verdict.level != after.verdict.level,
    )
