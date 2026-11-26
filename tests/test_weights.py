import numpy as np
import pytest
from pydantic import ValidationError

from quanheng.weights import (
    PairwiseJudgment as J,
)
from quanheng.weights import (
    WeightSpec,
    noise_factors,
    rescale_for_anchor_change,
    weights_from_pairwise,
)


def test_consistent_pairwise():
    r = weights_from_pairwise(
        ["a", "b", "c"],
        [J(a="a", b="b", ratio=2), J(a="b", b="c", ratio=3), J(a="a", b="c", ratio=6)],
    )
    assert r.weights == pytest.approx({"a": 0.6, "b": 0.3, "c": 0.1})
    assert r.consistency_ratio == pytest.approx(0, abs=1e-9) and r.most_inconsistent is None


def test_inconsistent_pair_is_named():
    js = [
        J(a="a", b="b", ratio=2),
        J(a="a", b="c", ratio=4),
        J(a="a", b="d", ratio=8),
        J(a="b", b="c", ratio=2),
        J(a="b", b="d", ratio=4),
        J(a="c", b="d", ratio=0.25),
    ]
    r = weights_from_pairwise(list("abcd"), js)
    assert r.consistency_ratio == pytest.approx(0.2118, abs=1e-3)
    assert r.most_inconsistent == ("c", "d")
    assert r.weights == pytest.approx(
        {"a": 0.5380, "b": 0.2690, "c": 0.0800, "d": 0.1131}, abs=1e-3
    )


def test_reversed_judgment_and_two_criteria():
    r1 = weights_from_pairwise(["a", "b"], [J(a="a", b="b", ratio=2)])
    r2 = weights_from_pairwise(["a", "b"], [J(a="b", b="a", ratio=0.5)])
    assert r1.weights == pytest.approx({"a": 2 / 3, "b": 1 / 3}) == r2.weights
    assert r1.consistency_ratio == 0


def test_incomplete_or_duplicate_judgments_rejected():
    with pytest.raises(ValueError):
        weights_from_pairwise(["a", "b", "c"], [J(a="a", b="b", ratio=2)])
    with pytest.raises(ValueError):
        weights_from_pairwise(["a", "b"], [J(a="a", b="b", ratio=2), J(a="b", b="a", ratio=1)])


def test_other_invalid_judgments_rejected():
    with pytest.raises(ValueError):
        weights_from_pairwise(["a", "b"], [J(a="a", b="z", ratio=2)])
    with pytest.raises(ValueError):
        weights_from_pairwise(["a", "b"], [J(a="a", b="a", ratio=2)])
    with pytest.raises(ValidationError):
        J(a="a", b="b", ratio=10)
    assert J(a="a", b="b", ratio=9).ratio == 9
    many = [f"c{i}" for i in range(11)]
    js = [J(a=x, b=y, ratio=1) for i, x in enumerate(many) for y in many[i + 1 :]]
    with pytest.raises(ValueError):
        weights_from_pairwise(many, js)


@pytest.mark.parametrize(
    ("certainty", "cr", "expected"),
    [
        ("fixed", None, 0.0),
        ("high", None, 0.1),
        ("medium", None, 0.25),
        ("low", None, 0.5),
        ("medium", 0.1, 0.25),
        ("medium", 0.2, 0.375),
    ],
)
def test_sigma(certainty, cr, expected):
    w = WeightSpec(values={"a": 1}, certainty=certainty, consistency_ratio=cr)
    assert w.sigma() == pytest.approx(expected)


def test_normalized_and_validation():
    assert WeightSpec(values={"a": 30, "b": 10}).normalized() == {"a": 0.75, "b": 0.25}
    for bad in ({"a": -1, "b": 2}, {"a": 0, "b": 0}, {}):
        with pytest.raises(ValidationError):
            WeightSpec(values=bad)


def test_rescale_keeps_exchange_rate():
    w = rescale_for_anchor_change(
        {"rent": 0.30, "commute": 0.35, "light": 0.15, "size": 0.20},
        "rent",
        old=(90000, 60000),
        new=(100000, 55000),
    )
    assert w == pytest.approx(
        {"rent": 0.391304, "commute": 0.304348, "light": 0.130435, "size": 0.173913}, abs=1e-6
    )
    assert list(w) == ["rent", "commute", "light", "size"]
    with pytest.raises(ValueError):  # 方向反了
        rescale_for_anchor_change({"rent": 1.0}, "rent", old=(90000, 60000), new=(55000, 100000))


def test_noise_factors():
    assert np.array_equal(noise_factors(0, ["a", "b"], 0.0, 100), np.ones((100, 2)))
    f = noise_factors(0, ["a", "b"], 0.25, 1000)
    assert f.shape == (1000, 2) and np.all(f > 0) and np.all(np.isfinite(f))
    assert np.array_equal(f[:, 1], noise_factors(0, ["b"], 0.25, 1000)[:, 0])  # 各准则的流互不影响
