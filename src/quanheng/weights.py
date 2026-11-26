import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator
from scipy.special import ndtri

from quanheng._base import Model
from quanheng.sampling import quantiles

Certainty = Literal["fixed", "high", "medium", "low"]
SIGMA: dict[Certainty, float] = {"fixed": 0.0, "high": 0.1, "medium": 0.25, "low": 0.5}
POOR_CONSISTENCY = 0.1  # a consistency ratio above this counts as poor
POOR_CONSISTENCY_WIDENING = 1.5

_RANDOM_INDEX = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
_TIE_TOLERANCE = 1e-12


class PairwiseJudgment(Model):
    a: str
    b: str
    ratio: float = Field(ge=1 / 9, le=9)  # a's swing is `ratio` times as important as b's


class PairwiseResult(Model):
    weights: dict[str, float]
    consistency_ratio: float
    most_inconsistent: tuple[str, str] | None


class WeightSpec(Model):
    values: dict[str, float]
    certainty: Certainty = "medium"
    consistency_ratio: float | None = None
    pairwise: list[PairwiseJudgment] | None = None  # record only, never used in computation

    @field_validator("values")
    @classmethod
    def _check_values(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("weights must not be empty")
        if any(x < 0 for x in v.values()):
            raise ValueError("weights must be >= 0")
        if sum(v.values()) <= 0:
            raise ValueError("weights must sum to more than 0")
        return v

    def normalized(self) -> dict[str, float]:
        total = sum(self.values.values())
        return {k: x / total for k, x in self.values.items()}

    def sigma(self) -> float:
        s = SIGMA[self.certainty]
        if self.consistency_ratio is not None and self.consistency_ratio > POOR_CONSISTENCY:
            s *= POOR_CONSISTENCY_WIDENING
        return s


def weights_from_pairwise(criteria: list[str], judgments: list[PairwiseJudgment]) -> PairwiseResult:
    n = len(criteria)
    if n > 10:
        raise ValueError("at most 10 criteria can be compared pairwise")
    index = {c: i for i, c in enumerate(criteria)}
    matrix = np.ones((n, n), dtype=np.float64)
    seen: set[frozenset[str]] = set()
    for item in judgments:
        if item.a not in index or item.b not in index:
            raise ValueError(f"judgment names an unknown criterion: {item.a!r} vs {item.b!r}")
        if item.a == item.b:
            raise ValueError(f"judgment compares {item.a!r} with itself")
        pair = frozenset((item.a, item.b))
        if pair in seen:
            raise ValueError(f"pair {item.a!r}, {item.b!r} is judged more than once")
        seen.add(pair)
        matrix[index[item.a], index[item.b]] = item.ratio
        matrix[index[item.b], index[item.a]] = 1 / item.ratio
    if len(seen) != n * (n - 1) // 2:
        raise ValueError("every pair of criteria must be judged exactly once")

    w = np.exp(np.log(matrix).mean(axis=1))
    w = w / w.sum()
    weights = {c: float(w[i]) for c, i in index.items()}
    if n <= 2:
        return PairwiseResult(weights=weights, consistency_ratio=0.0, most_inconsistent=None)

    lam = float(np.mean((matrix @ w) / w))
    cr = (lam - n) / (n - 1) / _RANDOM_INDEX[n]
    worst: tuple[str, str] | None = None
    if cr > POOR_CONSISTENCY:
        best = -1.0
        for i in range(n):
            for j in range(i + 1, n):
                err = abs(math.log(matrix[i, j] * w[j] / w[i]))
                if err > best + _TIE_TOLERANCE:
                    best, worst = err, (criteria[i], criteria[j])
    return PairwiseResult(weights=weights, consistency_ratio=cr, most_inconsistent=worst)


def rescale_for_anchor_change(
    weights: dict[str, float],
    criterion: str,
    old: tuple[float, float],
    new: tuple[float, float],
) -> dict[str, float]:
    """Keep the exchange rate when a criterion's (bad, good) anchors move."""
    factor = (new[1] - new[0]) / (old[1] - old[0])
    if not factor > 0:
        raise ValueError("anchors must keep their direction (good stays on the same side of bad)")
    scaled = {k: (x * factor if k == criterion else x) for k, x in weights.items()}
    total = sum(scaled.values())
    return {k: x / total for k, x in scaled.items()}


def noise_factors(seed: int, criteria: list[str], sigma: float, n: int) -> NDArray[np.float64]:
    out = np.ones((n, len(criteria)), dtype=np.float64)
    if sigma == 0:
        return out
    lo, hi = 2.0**-53, 1 - 2.0**-53
    for j, cid in enumerate(criteria):
        u = np.clip(quantiles(seed, f"weight:{cid}", n), lo, hi)
        out[:, j] = np.exp(sigma * ndtri(u))
    return out
