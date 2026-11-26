import numpy as np
from numpy.typing import NDArray

from quanheng.choice.evaluate import Draws, Outcome
from quanheng.choice.model import ChoiceModel
from quanheng.choice.result import InfoValue

BINS = 10


def _one(target: str, y: NDArray[np.float64], u: NDArray[np.float64], total: float) -> InfoValue:
    mean = float(y.mean())
    groups = np.minimum(np.floor(u * BINS).astype(np.int64), BINS - 1)
    n = y.shape[0]
    between = 0.0
    rates: list[float | None] = []
    for g in range(BINS):
        mask = groups == g
        count = int(mask.sum())
        if count == 0:
            rates.append(None)
            continue
        group_mean = float(y[mask].mean())
        rates.append(group_mean)
        between += (count / n) * (group_mean - mean) ** 2
    low = rates[0] if rates[0] is not None else mean
    high = rates[-1] if rates[-1] is not None else mean
    return InfoValue(
        target=target,
        share=float(between / total),
        leader_win_rate_low=float(low),
        leader_win_rate_high=float(high),
    )


def info_values(model: ChoiceModel, draws: Draws, outcome: Outcome, winner: str) -> list[InfoValue]:
    """Rank uncertain inputs by the share of the leader's win/lose variation they explain."""
    ids = [o.id for o in model.options]
    y = outcome.share[:, ids.index(winner)]
    total = float(np.var(y))
    if total == 0.0:
        return []

    found: list[InfoValue] = []
    for i, o in enumerate(model.options):
        for j, c in enumerate(model.criteria):
            if model.scores[o.id][c.id].is_uncertain:
                found.append(_one(f"scores.{o.id}.{c.id}", y, draws.quantile[:, i, j], total))
    if len(model.criteria) > 1 and model.weights.sigma() > 0:
        for j, c in enumerate(model.criteria):
            found.append(_one(f"weights.{c.id}", y, draws.weight_quantile[:, j], total))
    return sorted(found, key=lambda v: -v.share)
