import numpy as np
import pytest

from quanheng.choice.evaluate import draw
from quanheng.choice.model import ChoiceModel
from quanheng.choice.solve import solve_choice
from quanheng.choice.tipping import leader, leader_with_pinned


def by_target(result):
    return {t.target: t for t in result.tipping_points}


def test_hand_model_tipping_points(hand_model):
    tp = by_target(solve_choice(hand_model))
    assert tp["scores.b.commute"].flips_at == pytest.approx(40, abs=1e-3)
    assert tp["scores.b.commute"].new_leader == "b" and tp["scores.b.commute"].current == 50
    assert tp["scores.a.rent"].flips_at == pytest.approx(80000, abs=0.1)
    assert tp["scores.a.commute"].flips_at == pytest.approx(40, abs=1e-3)
    assert tp["scores.b.rent"].flips_at is None and tp["scores.b.rent"].new_leader is None
    assert tp["weights.rent"].flips_at == pytest.approx(2 / 3, abs=1e-6)
    assert tp["weights.commute"].flips_at == pytest.approx(1 / 3, abs=1e-6)


def test_crossing_a_tipping_point_changes_the_leader(rent_model):
    result = solve_choice(rent_model)
    draws = draw(rent_model, 10_000, 0)
    checked = 0
    for t in result.tipping_points:
        if t.flips_at is None or t.flips_at == t.current:
            continue
        step = (t.flips_at - t.current) * 1e-3
        assert leader_with_pinned(rent_model, draws, t.target, t.flips_at - step) == "a"
        new = leader_with_pinned(rent_model, draws, t.target, t.flips_at + step)
        assert new == t.new_leader != "a"
        checked += 1
    assert checked >= 8


def test_stated_range_and_ordering(rent_model):
    tps = solve_choice(rent_model).tipping_points
    tp = {t.target: t for t in tps}
    assert tp["scores.b.commute"].flips_at == pytest.approx(41.11, abs=0.01)
    assert tp["scores.b.commute"].within_stated_range is True  # 填的是 40–55
    assert tp["scores.a.commute"].within_stated_range is False  # 约 36.2，填的是 25–35
    assert tp["weights.light"].flips_at is None
    assert {t.target for t in tps[:3]} == {"weights.commute", "weights.rent", "scores.b.commute"}
    assert tps[-1].flips_at is None


def test_pinning_does_not_mutate_draws(hand_model):
    draws = draw(hand_model, 200, 0)
    before = draws.raw.copy()
    leader_with_pinned(hand_model, draws, "scores.a.rent", 1.0)
    assert (draws.raw == before).all()


def test_bad_target_raises(hand_model):
    draws = draw(hand_model, 200, 0)
    for bad in ["scores.a", "scores.z.rent", "weights.nope", "foo.bar"]:
        with pytest.raises(ValueError):
            leader_with_pinned(hand_model, draws, bad, 1.0)


def test_single_criterion_and_extreme_weights():
    base = {"title": "t", "options": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]}
    one = ChoiceModel.model_validate(
        {
            **base,
            "criteria": [{"id": "q", "name": "Q", "bad": 0, "good": 10}],
            "scores": {"a": {"q": 6}, "b": {"q": [3, 5]}},
            "weights": {"values": {"q": 1}},
        }
    )
    targets = {t.target for t in solve_choice(one).tipping_points}
    assert targets == {"scores.a.q", "scores.b.q"}  # 没有权重目标
    two = ChoiceModel.model_validate(
        {
            **base,
            "criteria": [
                {"id": "x", "name": "X", "bad": 0, "good": 10},
                {"id": "y", "name": "Y", "bad": 0, "good": 10},
            ],
            "scores": {"a": {"x": 6, "y": 1}, "b": {"x": 4, "y": 9}},
            "weights": {"values": {"x": 1, "y": 0}},
        }
    )
    tp = {t.target: t for t in solve_choice(two).tipping_points}
    assert "weights.x" not in tp  # 权重已经是 1，跳过
    assert tp["weights.y"].flips_at == pytest.approx(0.2, abs=0.02)


def budget_model():
    """B is always over budget, so A wins every sample while A's rent stays within it."""
    return ChoiceModel.model_validate(
        {
            "title": "t",
            "options": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            "criteria": [
                {"id": "rent", "name": "房租", "unit": "日元/月", "bad": 90000, "good": 60000},
                {"id": "q", "name": "Q", "bad": 0, "good": 10},
            ],
            "constraints": [{"criterion": "rent", "op": "<=", "threshold": 85000}],
            "scores": {"a": {"rent": 78000, "q": [4, 6]}, "b": {"rent": 88000, "q": [5, 9]}},
            "weights": {"values": {"rent": 1, "q": 1}, "certainty": "fixed"},
        }
    )


def test_leader_rule_counts_nothing_feasible_as_an_outcome():
    rates = np.array([0.0, 0.0])
    assert leader(rates, 1.0) is None
    assert leader(np.array([0.3, 0.3]), 0.4) is None
    assert leader(np.array([0.4, 0.4]), 0.2) == 0  # 并列：先声明的候选
    assert leader(np.array([0.2, 0.45]), 0.35) == 1
    assert leader(np.array([0.5, 0.0]), 0.5) == 0  # 一样多时不算"没有可行方案"


def test_pinning_into_nothing_feasible_is_a_flip():
    m = budget_model()
    result = solve_choice(m)
    assert result.verdict.winner == "a" and result.win_rates["a"] == 1.0
    tp = by_target(result)["scores.a.rent"]
    assert tp.flips_at == pytest.approx(85000, abs=0.01)
    assert tp.new_leader is None
    draws = draw(m, 10_000, 0)
    assert leader_with_pinned(m, draws, "scores.a.rent", 86000) is None
    assert leader_with_pinned(m, draws, "scores.a.rent", 84000) == "a"
