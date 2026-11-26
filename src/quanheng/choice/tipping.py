import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quanheng.choice.evaluate import Draws, central_weights, evaluate
from quanheng.choice.model import ChoiceModel
from quanheng.choice.result import TippingPoint

GRID_POINTS = 101
BISECTION_STEPS = 40


def leading_index(rates: NDArray[np.float64]) -> int:
    """Highest win rate; ties go to the earlier-declared option."""
    return int(np.argsort(-rates, kind="stable")[0])


def leader_with_pinned(model: ChoiceModel, draws: Draws, target: str, value: float) -> str:
    """The option with the highest win rate when `target` is pinned at `value`."""
    option_ids = [o.id for o in model.options]
    criterion_ids = [c.id for c in model.criteria]
    parts = target.split(".")
    raw = draws.raw
    weights = central_weights(model)
    if (
        len(parts) == 3
        and parts[0] == "scores"
        and parts[1] in option_ids
        and parts[2] in criterion_ids
    ):
        raw = draws.raw.copy()
        raw[:, option_ids.index(parts[1]), criterion_ids.index(parts[2])] = value
    elif len(parts) == 2 and parts[0] == "weights" and parts[1] in criterion_ids:
        j = criterion_ids.index(parts[1])
        original = float(weights[j])
        scaled = weights * ((1.0 - value) / (1.0 - original))
        scaled[j] = value
        weights = scaled
    else:
        raise ValueError(f"unknown tipping target: {target!r}")
    outcome = evaluate(model, raw, weights, draws.weight_factor)
    return option_ids[leading_index(outcome.share.mean(axis=0))]


@dataclass(frozen=True)
class _Target:
    name: str
    current: float
    lo: float
    hi: float
    stated: tuple[float, float]


def _targets(model: ChoiceModel) -> list[_Target]:
    out: list[_Target] = []
    for o in model.options:
        for c in model.criteria:
            q = model.scores[o.id][c.id]
            out.append(
                _Target(
                    f"scores.{o.id}.{c.id}",
                    q.central,
                    min(c.bad, c.good),
                    max(c.bad, c.good),
                    q.bounds,
                )
            )
    if len(model.criteria) > 1:
        sigma = model.weights.sigma()
        for c, w in zip(model.criteria, central_weights(model), strict=True):
            w = float(w)
            if w < 1.0:
                out.append(
                    _Target(
                        f"weights.{c.id}",
                        w,
                        0.0,
                        1.0,
                        (w * math.exp(-2 * sigma), w * math.exp(2 * sigma)),
                    )
                )
    return out


def _solve_target(
    model: ChoiceModel, draws: Draws, winner: str, t: _Target
) -> tuple[float | None, str | None]:
    def leader(v: float) -> str:
        return leader_with_pinned(model, draws, t.name, v)

    at_current = leader(t.current)
    if at_current != winner:
        return t.current, at_current
    grid = np.linspace(t.lo, t.hi, GRID_POINTS)
    flipped = [float(v) for v in grid if leader(float(v)) != winner]
    if not flipped:
        return None, None
    far = min(flipped, key=lambda v: abs(v - t.current))
    good, bad = t.current, far
    for _ in range(BISECTION_STEPS):
        mid = (good + bad) / 2
        if leader(mid) == winner:
            good = mid
        else:
            bad = mid
    return (good + bad) / 2, leader(bad)


def tipping_points(model: ChoiceModel, draws: Draws, winner: str) -> list[TippingPoint]:
    """For each input, where the leader would change (same criterion as the verdict)."""
    found: list[tuple[TippingPoint, float]] = []
    for t in _targets(model):
        flips_at, new_leader = _solve_target(model, draws, winner, t)
        within = flips_at is not None and t.stated[0] <= flips_at <= t.stated[1]
        distance = math.inf if flips_at is None else abs(flips_at - t.current) / (t.hi - t.lo)
        found.append(
            (
                TippingPoint(
                    target=t.name,
                    current=t.current,
                    flips_at=flips_at,
                    new_leader=new_leader,
                    within_stated_range=within,
                ),
                distance,
            )
        )
    found.sort(key=lambda item: (not item[0].within_stated_range, item[1]))
    return [tp for tp, _ in found]
