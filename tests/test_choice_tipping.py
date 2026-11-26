import pytest

from quanheng.choice.evaluate import draw
from quanheng.choice.model import ChoiceModel
from quanheng.choice.solve import solve_choice
from quanheng.choice.tipping import leader_with_pinned


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
