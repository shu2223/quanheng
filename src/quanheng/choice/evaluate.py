from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quanheng.choice.model import ChoiceModel
from quanheng.errors import UnconfirmedInputError
from quanheng.sampling import from_quantiles, quantiles
from quanheng.weights import noise_factors

TIE_TOLERANCE = 1e-12
MIN_SAMPLES, MAX_SAMPLES = 100, 1_000_000


@dataclass(frozen=True)
class Draws:
    raw: NDArray[np.float64]  # (samples, options, criteria): the drawn raw values
    quantile: NDArray[np.float64]  # same shape: the quantiles behind them
    weight_factor: NDArray[np.float64]  # (samples, criteria): random weight factors
    weight_quantile: NDArray[np.float64]  # (samples, criteria): quantiles behind the factors


@dataclass(frozen=True)
class Outcome:
    scores: NDArray[np.float64]  # (samples, options)
    feasible: NDArray[np.bool_]  # (samples, options)
    share: NDArray[np.float64]  # (samples, options); rows sum to 1, or 0 if nobody is feasible
    none_feasible: NDArray[np.bool_]  # (samples,)


def central_weights(model: ChoiceModel) -> NDArray[np.float64]:
    normalized = model.weights.normalized()
    return np.array([normalized[c.id] for c in model.criteria], dtype=np.float64)


def draw(model: ChoiceModel, samples: int, seed: int) -> Draws:
    unconfirmed = model.unconfirmed_inputs()
    if unconfirmed:
        raise UnconfirmedInputError(unconfirmed)
    if not MIN_SAMPLES <= samples <= MAX_SAMPLES:
        raise ValueError(f"samples must be between {MIN_SAMPLES} and {MAX_SAMPLES}")
    shape = (samples, len(model.options), len(model.criteria))
    raw = np.empty(shape, dtype=np.float64)
    quantile = np.empty(shape, dtype=np.float64)
    for i, option in enumerate(model.options):
        for j, criterion in enumerate(model.criteria):
            u = quantiles(seed, f"score:{option.id}:{criterion.id}", samples)
            quantile[:, i, j] = u
            raw[:, i, j] = from_quantiles(model.scores[option.id][criterion.id], u)
    ids = [c.id for c in model.criteria]
    weight_factor = noise_factors(seed, ids, model.weights.sigma(), samples)
    weight_quantile = np.stack([quantiles(seed, f"weight:{cid}", samples) for cid in ids], axis=1)
    return Draws(raw, quantile, weight_factor, weight_quantile)


def evaluate(
    model: ChoiceModel,
    raw: NDArray[np.float64],
    weights: NDArray[np.float64],
    weight_factor: NDArray[np.float64],
) -> Outcome:
    bad = np.array([c.bad for c in model.criteria], dtype=np.float64)
    good = np.array([c.good for c in model.criteria], dtype=np.float64)
    capped = np.array([c.cap_at_good for c in model.criteria], dtype=np.bool_)
    value = (raw - bad) / (good - bad)
    value = np.where(capped, np.minimum(value, 1.0), value)

    w = weights * weight_factor
    w = w / w.sum(axis=1, keepdims=True)
    scores = np.einsum("soc,sc->so", value, w)

    feasible = np.ones(scores.shape, dtype=np.bool_)
    index = {c.id: j for j, c in enumerate(model.criteria)}
    for k in model.constraints:
        x = raw[:, :, index[k.criterion]]
        feasible &= (x <= k.threshold) if k.op == "<=" else (x >= k.threshold)

    masked = np.where(feasible, scores, -np.inf)
    best = masked.max(axis=1, keepdims=True)
    tied = feasible & (scores >= best - TIE_TOLERANCE)
    count = tied.sum(axis=1, keepdims=True)
    share = np.where(tied, 1.0 / np.maximum(count, 1), 0.0).astype(np.float64)
    none_feasible = np.asarray(~feasible.any(axis=1), dtype=np.bool_)
    return Outcome(scores, feasible, share, none_feasible)
