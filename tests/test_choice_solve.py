import numpy
import pytest

import quanheng
from quanheng.choice.model import ChoiceModel
from quanheng.choice.result import Thresholds
from quanheng.choice.solve import solve_choice


def patched(model, fn):
    data = model.model_dump()
    fn(data)
    return ChoiceModel.model_validate(data)


def one_criterion(a, b, certainty="fixed"):
    return ChoiceModel.model_validate(
        {
            "title": "t",
            "options": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            "criteria": [{"id": "q", "name": "Q", "bad": 0, "good": 10}],
            "scores": {"a": {"q": a}, "b": {"q": b}},
            "weights": {"values": {"q": 1}, "certainty": certainty},
        }
    )


def test_hand_model_result(hand_model):
    r = solve_choice(hand_model)
    assert (r.verdict.level, r.verdict.winner, r.verdict.runner_up) == ("clear", "a", "b")
    assert r.verdict.win_rate == 1.0 and r.verdict.std_error == 0.0
    assert r.win_rates == {"a": 1.0, "b": 0.0} and r.none_feasible_share == 0.0
    assert r.feasibility == {"a": 1.0, "b": 1.0}
    assert r.central_scores == pytest.approx({"a": 0.625, "b": 0.5})
    assert r.contributions["a"] == pytest.approx({"rent": 0.25, "commute": 0.375})
    assert r.contributions["b"] == pytest.approx({"rent": 0.375, "commute": 0.125})
    assert r.central_leader == "a" and r.warnings == [] and r.constraint_gaps == []
    assert r.margin.p50 == pytest.approx(0.125) and r.margin.used_share == 1.0
    assert len(r.margin.counts) == 20 and sum(r.margin.counts) == 10_000
    assert r.meta.model_dump() == {
        "seed": 0,
        "samples": 10_000,
        "engine_version": quanheng.__version__,
        "numpy_version": numpy.__version__,
    }
    assert r.assumptions == [
        "independent_inputs",
        "linear_value",
        "uniform_range",
        "triangular_three_point",
        "proportional_weight_noise",
    ]


def test_known_win_rate_and_thresholds():
    m = one_criterion(5, [0, 50 / 7])  # B 超过 5 的概率正好是 30%
    r = solve_choice(m)
    assert r.win_rates["a"] == pytest.approx(0.70, abs=0.015) and r.verdict.level == "leaning"
    assert r.verdict.std_error == pytest.approx(0.0046, abs=2e-4)
    assert solve_choice(m, thresholds=Thresholds(clear=0.6)).verdict.level == "clear"
    assert solve_choice(m, thresholds=Thresholds(leaning=0.75)).verdict.level == "toss_up"


def test_win_rates_and_none_feasible_sum_to_one(rent_model):
    r = solve_choice(rent_model)
    assert sum(r.win_rates.values()) + r.none_feasible_share == pytest.approx(1.0)
    assert r.verdict.level == "leaning" and r.verdict.winner == "a"
    assert 0.78 < r.verdict.win_rate < 0.84


def test_mostly_infeasible(hand_model):
    m = patched(
        hand_model,
        lambda d: d.update(constraints=[{"criterion": "rent", "op": "<=", "threshold": 50000}]),
    )
    r = solve_choice(m)
    assert r.verdict.level == "mostly_infeasible" and r.verdict.winner is None
    assert r.none_feasible_share == 1.0 and r.margin is None and r.central_leader is None
    assert [g.model_dump() for g in r.constraint_gaps] == [
        {"index": 0, "criterion": "rent", "nearest_option": "b", "gap": 10000.0}
    ]
    assert r.tipping_points == [] and r.info_values == []


def test_same_inputs_give_identical_bytes(rent_model):
    assert solve_choice(rent_model).model_dump_json() == solve_choice(rent_model).model_dump_json()
    assert solve_choice(rent_model, seed=1).win_rates != solve_choice(rent_model).win_rates


def test_outside_anchor_warning(hand_model):
    r = solve_choice(patched(hand_model, lambda d: d["scores"]["a"].update(rent=95000)))
    assert [w.model_dump() for w in r.warnings] == [
        {"code": "outside_anchors", "target": "scores.a.rent"}
    ]


def test_central_values_can_disagree_with_sampling():
    r = solve_choice(one_criterion([0, 0, 10], 2))  # A 最可能是 0，但有 64% 的情况超过 2
    assert r.central_leader == "b" and r.verdict.winner == "a"
    assert r.win_rates["a"] == pytest.approx(0.64, abs=0.015)
    assert "central_disagrees" in [w.code for w in r.warnings]


def test_poor_consistency_warning(hand_model):
    r = solve_choice(patched(hand_model, lambda d: d["weights"].update(consistency_ratio=0.2)))
    assert "poor_consistency" in [w.code for w in r.warnings]
