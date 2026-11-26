import math

from quanheng import zh
from quanheng.choice.model import ChoiceModel, Criterion
from quanheng.choice.result import ChoiceResult, Thresholds, TippingPoint
from quanheng.quantity import Exact, Quantity, Range
from quanheng.whatif import WhatIfResult

MAX_TIPPING = 5
MAX_INFO = 3
MIN_INFO_SHARE = 0.01


def fmt_number(x: float) -> str:
    r = float(f"{x:.4g}")
    if abs(r) >= 1000:
        return f"{round(r):,}"
    if r.is_integer():
        return str(int(r))
    return f"{r:g}"


def _whole_percent(p: float) -> int:
    return int(f"{p * 100:.0f}")


def fmt_percent(p: float) -> str:
    """A share as a whole percent that never rounds onto 0% or 100% unless it is exactly that."""
    if p == 0:
        return "0%"
    if p == 1:
        return "100%"
    if p < 0.005:
        return "<1%"
    if p >= 0.995:
        return ">99%"
    # 0.005 * 100 is exactly 0.5 in binary, which rounds half to even, i.e. to 0.
    return f"{max(_whole_percent(p), 1)}%"


def _verdict_percent(p: float, thresholds: Thresholds) -> str:
    """The verdict's win rate; never rounded up onto a level threshold that p is below."""
    shown = _whole_percent(p)
    for threshold in (thresholds.clear, thresholds.leaning):
        if p < threshold <= shown / 100:
            return f"{math.floor(p * 1000) / 10:.1f}%"
    return fmt_percent(p)


def _unit(unit: str) -> str:
    return f" {unit}" if unit else ""


def _option_name(model: ChoiceModel, option_id: str) -> str:
    return next(o.name for o in model.options if o.id == option_id)


def _criterion(model: ChoiceModel, criterion_id: str) -> Criterion:
    return next(c for c in model.criteria if c.id == criterion_id)


def _numbers_text(nums: list[float]) -> str:
    if len(nums) == 1:
        return fmt_number(nums[0])
    if len(nums) == 2:
        return zh.RANGE_TEXT.format(low=fmt_number(nums[0]), high=fmt_number(nums[1]))
    return zh.THREE_POINT_TEXT.format(
        low=fmt_number(nums[0]), mode=fmt_number(nums[1]), high=fmt_number(nums[2])
    )


def _quantity_text(q: Quantity) -> str:
    v = q.value
    if isinstance(v, Exact):
        return _numbers_text([v.value])
    if isinstance(v, Range):
        return _numbers_text([v.low, v.high])
    return _numbers_text([v.low, v.mode, v.high])


def _verdict_line(model: ChoiceModel, result: ChoiceResult) -> str:
    v = result.verdict
    if v.level == "mostly_infeasible" or v.winner is None:
        return zh.VERDICT_INFEASIBLE.format(p=fmt_percent(result.none_feasible_share))
    err = f"±{1.96 * v.std_error * 100:.1f}%"
    template = zh.VERDICT_TOSS_UP if v.level == "toss_up" else zh.VERDICT_DECISIVE
    return template.format(
        level=zh.LEVEL_NAMES[v.level],
        winner=_option_name(model, v.winner),
        p=_verdict_percent(v.win_rate, result.meta.thresholds),
        err=err,
    )


def _conclusion_block(model: ChoiceModel, result: ChoiceResult) -> list[str]:
    lines = [model.title, _verdict_line(model, result), zh.REMINDER]
    if result.none_feasible_share > 0 and result.verdict.level != "mostly_infeasible":
        lines.append(zh.SOME_INFEASIBLE.format(p=fmt_percent(result.none_feasible_share)))
    for o in model.options:
        share = result.feasibility.get(o.id, 1.0)
        if share < 1:
            lines.append(zh.OPTION_INFEASIBLE.format(option=o.name, p=fmt_percent(1 - share)))
    return lines


def _why_block(model: ChoiceModel, result: ChoiceResult) -> list[str]:
    lines: list[str] = []
    winner_id = result.verdict.winner
    second_id = result.verdict.runner_up
    if winner_id is None or second_id is None:
        return lines
    diffs = [
        (c.name, result.contributions[winner_id][c.id] - result.contributions[second_id][c.id])
        for c in model.criteria
    ]
    diffs.sort(key=lambda t: -abs(t[1]))  # stable on criterion order
    lines.append(
        zh.CONTRIB_TITLE.format(
            leader=_option_name(model, winner_id), second=_option_name(model, second_id)
        )
    )
    lines.extend(zh.CONTRIB_LINE.format(criterion=n, diff=f"{d:+.2f}") for n, d in diffs)
    central_id = result.central_leader
    if central_id is not None and any(w.code == "central_disagrees" for w in result.warnings):
        lines.append(zh.CENTRAL_DISAGREES.format(option=_option_name(model, central_id)))
    return lines


def _tipping_line(model: ChoiceModel, t: TippingPoint) -> str:
    assert t.flips_at is not None
    # Equal means pinning the input at its central value already flips: there is no direction.
    at_current = t.flips_at == t.current
    direction = zh.BELOW if t.flips_at < t.current else zh.ABOVE
    parts = t.target.split(".")
    if parts[0] == "weights":
        criterion = _criterion(model, parts[1]).name
        value = f"{t.flips_at:.2f}"
        if at_current:
            return zh.TIPPING_WEIGHT_AT_CURRENT.format(criterion=criterion, value=value)
        return zh.TIPPING_WEIGHT.format(
            criterion=criterion, direction=direction, value=value, current=f"{t.current:.2f}"
        )
    crit = _criterion(model, parts[2])
    option = _option_name(model, parts[1])
    value = fmt_number(t.flips_at)
    if at_current:
        return zh.TIPPING_SCORE_AT_CURRENT.format(
            option=option, criterion=crit.name, value=value, unit=_unit(crit.unit)
        )
    return zh.TIPPING_SCORE.format(
        option=option,
        criterion=crit.name,
        direction=direction,
        value=value,
        unit=_unit(crit.unit),
        stated=_quantity_text(model.scores[parts[1]][parts[2]]),
    )


def _tipping_block(model: ChoiceModel, result: ChoiceResult) -> list[str]:
    lines = [zh.TIPPING_TITLE]
    with_value = [t for t in result.tipping_points if t.flips_at is not None]
    without = len(result.tipping_points) - len(with_value)
    lines.extend(_tipping_line(model, t) for t in with_value[:MAX_TIPPING])
    hidden = len(with_value) - MAX_TIPPING
    if hidden > 0:
        lines.append(zh.TIPPING_MORE.format(k=hidden))
    if not with_value:
        lines.append(zh.TIPPING_NONE)
    elif without:
        lines.append(zh.TIPPING_REST.format(n=without))
    return lines


def _info_block(model: ChoiceModel, result: ChoiceResult) -> list[str]:
    lines = [zh.INFO_TITLE]
    winner = result.verdict.winner
    shown = [i for i in result.info_values if i.share >= MIN_INFO_SHARE][:MAX_INFO]
    if not shown or winner is None:
        lines.append(zh.INFO_NONE)
        return lines
    leader = _option_name(model, winner)
    for n, info in enumerate(shown, start=1):
        parts = info.target.split(".")
        low = fmt_percent(info.leader_win_rate_low)
        high = fmt_percent(info.leader_win_rate_high)
        if parts[0] == "weights":
            lines.append(
                zh.INFO_WEIGHT.format(
                    n=n,
                    criterion=_criterion(model, parts[1]).name,
                    leader=leader,
                    low=low,
                    high=high,
                )
            )
        else:
            lines.append(
                zh.INFO_SCORE.format(
                    n=n,
                    option=_option_name(model, parts[1]),
                    criterion=_criterion(model, parts[2]).name,
                    leader=leader,
                    low=low,
                    high=high,
                )
            )
    return lines


def _gap_block(model: ChoiceModel, result: ChoiceResult) -> list[str]:
    lines = [zh.GAP_TITLE]
    for g in result.constraint_gaps:
        constraint = model.constraints[g.index]
        crit = _criterion(model, g.criterion)
        lines.append(
            zh.GAP_LINE.format(
                criterion=crit.name,
                op=constraint.op,
                threshold=fmt_number(constraint.threshold),
                option=_option_name(model, g.nearest_option),
                gap=fmt_number(g.gap),
                unit=_unit(crit.unit),
            )
        )
    return lines


def _notes_block(model: ChoiceModel, result: ChoiceResult) -> list[str]:
    lines: list[str] = []
    for w in result.warnings:
        if w.code == "outside_anchors" and w.target is not None:
            parts = w.target.split(".")
            lines.append(
                zh.OUTSIDE_ANCHORS.format(
                    option=_option_name(model, parts[1]),
                    criterion=_criterion(model, parts[2]).name,
                )
            )
    if any(w.code == "poor_consistency" for w in result.warnings):
        lines.append(zh.POOR_CONSISTENCY_NOTE)
    return lines


def explain_choice(model: ChoiceModel, result: ChoiceResult) -> str:
    blocks = [_conclusion_block(model, result)]
    if result.verdict.level == "mostly_infeasible":
        blocks.append(_gap_block(model, result))
    else:
        blocks.append(_why_block(model, result))
        blocks.append(_tipping_block(model, result))
        blocks.append(_info_block(model, result))
    blocks.append(_notes_block(model, result))
    blocks.append([zh.ASSUMPTIONS_TITLE, *(zh.ASSUMPTION_PREFIX + a for a in zh.ASSUMPTION_LINES)])
    m = result.meta
    blocks.append(
        [
            zh.META_LINE.format(
                samples=fmt_number(m.samples),
                seed=str(m.seed),
                engine=m.engine_version,
                numpy=m.numpy_version,
            )
        ]
    )
    return "\n\n".join("\n".join(b) for b in blocks if b)


def _change_line(model: ChoiceModel, target: str, before: list[float], after: list[float]) -> str:
    parts = target.split(".")
    if parts[0] == "weights":
        name = zh.WHATIF_WEIGHT_TARGET.format(criterion=_criterion(model, parts[1]).name)
        b, a = f"{before[0]:.2f}", f"{after[0]:.2f}"
    elif parts[0] == "constraints":
        crit_id = model.constraints[int(parts[1])].criterion
        name = zh.WHATIF_CONSTRAINT_TARGET.format(criterion=_criterion(model, crit_id).name)
        b, a = fmt_number(before[0]), fmt_number(after[0])
    else:
        name = zh.WHATIF_SCORE_TARGET.format(
            option=_option_name(model, parts[1]), criterion=_criterion(model, parts[2]).name
        )
        b, a = _numbers_text(before), _numbers_text(after)
    return zh.WHATIF_CHANGE.format(target=name, before=b, after=a)


def explain_whatif(model: ChoiceModel, whatif: WhatIfResult) -> str:
    lines = [_change_line(model, c.target, c.before, c.after) for c in whatif.changes]
    lines.append(zh.WHATIF_BEFORE.format(verdict=_verdict_line(model, whatif.before)))
    lines.append(zh.WHATIF_AFTER.format(verdict=_verdict_line(model, whatif.after)))
    lines.append(zh.WHATIF_FLIPPED if whatif.winner_changed else zh.WHATIF_NOT_FLIPPED)
    return "\n".join(lines)
