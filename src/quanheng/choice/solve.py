import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import quanheng
from quanheng.choice.evaluate import TIE_TOLERANCE, central_weights, draw, evaluate
from quanheng.choice.model import ChoiceModel
from quanheng.choice.result import (
    ChoiceResult,
    ConstraintGap,
    Margin,
    Meta,
    ResultWarning,
    Thresholds,
    Verdict,
)
from quanheng.choice.tipping import leading_index, tipping_points
from quanheng.weights import POOR_CONSISTENCY

ASSUMPTIONS = [
    "independent_inputs",
    "linear_value",
    "uniform_range",
    "triangular_three_point",
    "proportional_weight_noise",
]

Level = Literal["clear", "leaning", "toss_up", "mostly_infeasible"]


def _central(
    model: ChoiceModel,
) -> tuple[dict[str, dict[str, float]], dict[str, float], str | None, NDArray[np.float64]]:
    """Central-value scores, using the same scoring rule as `evaluate`."""
    n_opt, n_crit = len(model.options), len(model.criteria)
    raw = np.array(
        [[model.scores[o.id][c.id].central for c in model.criteria] for o in model.options],
        dtype=np.float64,
    ).reshape(1, n_opt, n_crit)
    weights = central_weights(model)
    ones = np.ones((1, n_crit), dtype=np.float64)
    # A one-hot weight vector makes `evaluate` return the plain value of one criterion.
    values = np.empty((n_opt, n_crit), dtype=np.float64)
    for j in range(n_crit):
        unit = np.zeros(n_crit, dtype=np.float64)
        unit[j] = 1.0
        values[:, j] = evaluate(model, raw, unit, ones).scores[0]
    feasible = evaluate(model, raw, weights, ones).feasible[0]

    parts = values * weights
    contributions = {
        o.id: {c.id: float(parts[i, j]) for j, c in enumerate(model.criteria)}
        for i, o in enumerate(model.options)
    }
    totals = parts.sum(axis=1)
    central_scores = {o.id: float(totals[i]) for i, o in enumerate(model.options)}
    leader: str | None = None
    if feasible.any():
        best = float(totals[feasible].max())
        for i, o in enumerate(model.options):
            if feasible[i] and totals[i] >= best - TIE_TOLERANCE:
                leader = o.id
                break
    return contributions, central_scores, leader, raw[0]


def _constraint_gaps(model: ChoiceModel, central_raw: NDArray[np.float64]) -> list[ConstraintGap]:
    index = {c.id: j for j, c in enumerate(model.criteria)}
    gaps: list[ConstraintGap] = []
    for k, constraint in enumerate(model.constraints):
        column = central_raw[:, index[constraint.criterion]]
        if constraint.op == "<=":
            violation = np.maximum(0.0, column - constraint.threshold)
        else:
            violation = np.maximum(0.0, constraint.threshold - column)
        best = int(np.argmin(violation))  # first minimum: declared order wins ties
        gaps.append(
            ConstraintGap(
                index=k,
                criterion=constraint.criterion,
                nearest_option=model.options[best].id,
                gap=float(violation[best]),
            )
        )
    return gaps


def solve_choice(
    model: ChoiceModel,
    *,
    samples: int = 10_000,
    seed: int = 0,
    thresholds: Thresholds | None = None,
) -> ChoiceResult:
    thresholds = thresholds if thresholds is not None else Thresholds()
    draws = draw(model, samples, seed)
    outcome = evaluate(model, draws.raw, central_weights(model), draws.weight_factor)
    ids = [o.id for o in model.options]

    rates = outcome.share.mean(axis=0)
    win_rates = {oid: float(rates[i]) for i, oid in enumerate(ids)}
    none_share = float(outcome.none_feasible.mean())
    feasibility = {oid: float(outcome.feasible[:, i].mean()) for i, oid in enumerate(ids)}

    leader_i = leading_index(rates)
    second_i = int(np.argsort(-rates, kind="stable")[1])
    top = float(rates[leader_i])

    level: Level
    winner: str | None = None
    if none_share > top:
        level = "mostly_infeasible"
        verdict = Verdict(level=level, winner=None, runner_up=None, win_rate=0.0, std_error=0.0)
    else:
        if top >= thresholds.clear:
            level = "clear"
        elif top >= thresholds.leaning:
            level = "leaning"
        else:
            level = "toss_up"
        winner = ids[leader_i]
        verdict = Verdict(
            level=level,
            winner=winner,
            runner_up=ids[second_i],
            win_rate=top,
            std_error=math.sqrt(top * (1 - top) / samples),
        )

    margin: Margin | None = None
    if winner is not None:
        both = outcome.feasible[:, leader_i] & outcome.feasible[:, second_i]
        if both.any():
            diff = outcome.scores[both, leader_i] - outcome.scores[both, second_i]
            p5, p50, p95 = (float(q) for q in np.quantile(diff, [0.05, 0.5, 0.95]))
            counts, edges = np.histogram(diff, bins=20)
            margin = Margin(
                p5=p5,
                p50=p50,
                p95=p95,
                used_share=float(both.mean()),
                bin_edges=[float(e) for e in edges],
                counts=[int(c) for c in counts],
            )

    contributions, central_scores, central_leader, central_raw = _central(model)

    warnings: list[ResultWarning] = []
    for o in model.options:
        for c in model.criteria:
            low, high = model.scores[o.id][c.id].bounds
            if low < min(c.bad, c.good) or high > max(c.bad, c.good):
                warnings.append(
                    ResultWarning(code="outside_anchors", target=f"scores.{o.id}.{c.id}")
                )
    if winner is not None and winner != central_leader:
        warnings.append(ResultWarning(code="central_disagrees"))
    ratio = model.weights.consistency_ratio
    if ratio is not None and ratio > POOR_CONSISTENCY:
        warnings.append(ResultWarning(code="poor_consistency"))

    return ChoiceResult(
        meta=Meta(
            seed=seed,
            samples=samples,
            engine_version=quanheng.__version__,
            numpy_version=np.__version__,
        ),
        verdict=verdict,
        win_rates=win_rates,
        none_feasible_share=none_share,
        feasibility=feasibility,
        central_scores=central_scores,
        contributions=contributions,
        central_leader=central_leader,
        margin=margin,
        constraint_gaps=_constraint_gaps(model, central_raw),
        tipping_points=tipping_points(model, draws, winner) if winner is not None else [],
        warnings=warnings,
        assumptions=list(ASSUMPTIONS),
    )
