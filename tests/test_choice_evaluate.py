import numpy as np
import pytest

from quanheng.choice.evaluate import central_weights, draw, evaluate
from quanheng.choice.model import ChoiceModel
from quanheng.errors import UnconfirmedInputError
from quanheng.weights import rescale_for_anchor_change


def run(model, samples=10_000, seed=0):
    d = draw(model, samples, seed)
    return d, evaluate(model, d.raw, central_weights(model), d.weight_factor)


def patched(model, fn):
    data = model.model_dump()
    fn(data)
    return ChoiceModel.model_validate(data)


def test_hand_model_scores(hand_model):
    _, out = run(hand_model, samples=100)
    assert np.allclose(out.scores, [0.625, 0.5]) and np.all(out.share == [1.0, 0.0])


def test_identical_options_split_ties(hand_model):
    two = patched(hand_model, lambda d: d["scores"].update(b=d["scores"]["a"]))
    assert np.all(run(two, samples=100)[1].share == 0.5)
    three = patched(
        two,
        lambda d: (
            d["options"].append({"id": "c", "name": "C"}),
            d["scores"].update(c=d["scores"]["a"]),
        ),
    )
    assert np.allclose(run(three, samples=100)[1].share, 1 / 3)


def test_shares_plus_none_feasible_sum_to_one(rent_model):
    _, out = run(rent_model)
    assert np.allclose(out.share.sum(axis=1) + out.none_feasible, 1.0)


def test_constraints_are_judged_per_sample(hand_model):
    m = patched(
        hand_model,
        lambda d: (
            d["scores"]["a"].update(rent=[70000, 90000]),
            d.update(constraints=[{"criterion": "rent", "op": "<=", "threshold": 85000}]),
        ),
    )
    _, out = run(m)
    assert out.feasible[:, 0].mean() == pytest.approx(0.75, abs=0.015)
    assert out.feasible[:, 1].all()
    assert np.all(out.share[~out.feasible[:, 0], 0] == 0)


def test_nobody_feasible(hand_model):
    m = patched(
        hand_model,
        lambda d: d.update(constraints=[{"criterion": "rent", "op": "<=", "threshold": 50000}]),
    )
    _, out = run(m, samples=100)
    assert out.none_feasible.all() and np.all(out.share == 0)


def test_strict_dominance_always_wins(hand_model):
    m = patched(
        hand_model,
        lambda d: d["scores"].update(
            a={"rent": [55000, 60000], "commute": [20, 25]},
            b={"rent": [65000, 80000], "commute": [30, 55]},
        ),
    )
    m = patched(m, lambda d: d["weights"].update(certainty="low"))
    assert np.all(run(m)[1].share[:, 0] == 1.0)


def test_anchor_change_with_rescaled_weights_keeps_every_winner(rent_model):
    _, before = run(rent_model)

    def change(d):
        d["criteria"][0].update(bad=100000, good=55000)
        d["weights"]["values"] = rescale_for_anchor_change(
            rent_model.weights.normalized(), "rent", old=(90000, 60000), new=(100000, 55000)
        )

    _, after = run(patched(rent_model, change))
    assert np.array_equal(before.share, after.share)


def test_cap_at_good(hand_model):
    far = patched(
        hand_model,
        lambda d: (d["scores"]["a"].update(commute=10), d["criteria"][1].update(cap_at_good=True)),
    )
    at_good = patched(hand_model, lambda d: d["scores"]["a"].update(commute=20))
    assert run(far, 100)[1].scores[0, 0] == run(at_good, 100)[1].scores[0, 0]
    uncapped = patched(hand_model, lambda d: d["scores"]["a"].update(commute=10))
    assert run(uncapped, 100)[1].scores[0, 0] > run(at_good, 100)[1].scores[0, 0]


def test_changing_one_input_leaves_other_draws_untouched(rent_model):
    before = draw(rent_model, 1000, 0)
    after = draw(patched(rent_model, lambda d: d["scores"]["a"].update(commute=[30, 50])), 1000, 0)
    mask = np.ones(before.raw.shape[1:], bool)
    mask[0, 1] = False
    assert np.array_equal(before.raw[:, mask], after.raw[:, mask])
    assert np.array_equal(before.quantile, after.quantile)
    assert np.array_equal(before.weight_factor, after.weight_factor)


def test_unconfirmed_input_is_refused(rent_model):
    m = patched(rent_model, lambda d: d["scores"]["b"]["commute"].update(confirmed=False))
    with pytest.raises(UnconfirmedInputError) as e:
        draw(m, 1000, 0)
    assert e.value.targets == ["scores.b.commute"]


@pytest.mark.parametrize("samples", [0, -5, 99, 1_000_001])
def test_sample_count_bounds(hand_model, samples):
    with pytest.raises(ValueError):
        draw(hand_model, samples, 0)
