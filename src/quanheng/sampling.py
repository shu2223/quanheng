import hashlib

import numpy as np
from numpy.typing import NDArray

from quanheng.quantity import Exact, Quantity, Range


def quantiles(seed: int, key: str, n: int) -> NDArray[np.float64]:
    """n quantiles in [0, 1) from a stream determined only by (seed, key).

    The algorithm is pinned: changing it makes every saved result unreproducible.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    words = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, *words])))
    return rng.random(n)


def from_quantiles(q: Quantity, u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse CDF of the quantity's distribution evaluated at quantiles u."""
    v = q.value
    if isinstance(v, Exact):
        return np.full(u.shape, v.value, dtype=np.float64)
    if isinstance(v, Range):
        return np.asarray(v.low + u * (v.high - v.low), dtype=np.float64)
    span = v.high - v.low
    c = (v.mode - v.low) / span
    left = v.low + np.sqrt(u * span * (v.mode - v.low))
    right = v.high - np.sqrt((1 - u) * span * (v.high - v.mode))
    return np.asarray(np.where(u < c, left, right), dtype=np.float64)
