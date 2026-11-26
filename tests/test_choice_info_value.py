import pytest

from quanheng.choice.model import ChoiceModel
from quanheng.choice.solve import solve_choice


def test_decisive_and_irrelevant_inputs():
    m = ChoiceModel.model_validate(
        {
            "title": "t",
            "options": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            "criteria": [
                {"id": "x", "name": "X", "bad": 0, "good": 10},
                {"id": "y", "name": "Y", "bad": 0, "good": 10},
            ],
            "scores": {"a": {"x": 5, "y": [0, 10]}, "b": {"x": [0, 10], "y": [0, 10]}},
            "weights": {"values": {"x": 1, "y": 0}, "certainty": "fixed"},
        }
    )
    r = solve_choice(m)
    iv = {v.target: v for v in r.info_values}
    assert r.verdict.winner == "a" and r.info_values[0].target == "scores.b.x"
    assert iv["scores.b.x"].share == pytest.approx(1.0, abs=1e-6)
    assert iv["scores.b.x"].leader_win_rate_low == 1.0
    assert iv["scores.b.x"].leader_win_rate_high == 0.0
    assert iv["scores.a.y"].share < 0.01 and iv["scores.b.y"].share < 0.01
    assert not any(t.startswith("weights.") for t in iv)  # fixed weights do not take part


def test_nothing_to_learn_when_the_winner_never_loses(hand_model):
    assert solve_choice(hand_model).info_values == []


def test_example_ranks_the_guessed_commute_first(rent_model):
    ivs = solve_choice(rent_model).info_values
    first = ivs[0]
    assert first.target == "scores.b.commute" and 0.10 < first.share < 0.30
    assert first.leader_win_rate_low == pytest.approx(0.49, abs=0.05)
    assert first.leader_win_rate_high == pytest.approx(0.98, abs=0.02)
    assert "weights.commute" in {v.target for v in ivs}
    assert [v.share for v in ivs] == sorted((v.share for v in ivs), reverse=True)
