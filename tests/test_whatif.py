import numpy as np
import pytest

from quanheng.choice.evaluate import draw
from quanheng.errors import PatchError
from quanheng.whatif import PatchOp, apply_patch, compare


def op(target, kind, value):
    return PatchOp(target=target, op=kind, value=value)


def test_scale_shift_set_on_quantities(rent_model):
    m, changes = apply_patch(
        rent_model,
        [
            op("scores.a.rent", "scale", 1.2),
            op("scores.a.commute", "shift", 5),
            op("scores.b.commute", "set", 45),
            op("constraints.0.threshold", "shift", -10000),
        ],
    )
    assert [c.model_dump() for c in changes] == [
        {"target": "scores.a.rent", "before": [78000.0], "after": [93600.0]},
        {"target": "scores.a.commute", "before": [25.0, 35.0], "after": [30.0, 40.0]},
        {"target": "scores.b.commute", "before": [40.0, 55.0], "after": [45.0]},
        {"target": "constraints.0.threshold", "before": [85000.0], "after": [75000.0]},
    ]
    assert m.scores["b"]["commute"].source.type == "user_input"
    assert m.scores["a"]["commute"].source == rent_model.scores["a"]["commute"].source
    assert rent_model.scores["a"]["rent"].central == 78000  # input model untouched


def test_weight_patch_rescales_the_others(hand_model):
    m, changes = apply_patch(hand_model, [op("weights.rent", "set", 0.8)])
    assert m.weights.normalized() == pytest.approx({"rent": 0.8, "commute": 0.2})
    assert changes[0].before == [0.5] and changes[0].after == [0.8]


def test_weight_patch_keeps_other_spec_fields(hand_model):
    m, _ = apply_patch(hand_model, [op("weights.rent", "set", 0.8)])
    assert m.weights.certainty == hand_model.weights.certainty
    assert sum(m.weights.values.values()) == pytest.approx(1.0)


def test_compare_reports_a_flip(hand_model):
    r = compare(hand_model, [op("weights.rent", "set", 0.8)])
    assert r.before.verdict.winner == "a" and r.after.verdict.winner == "b"
    assert r.winner_changed and r.win_rate_delta == {"a": -1.0, "b": 1.0}
    assert r.level_changed is False


def test_rent_increase_breaks_the_budget(rent_model):
    r = compare(rent_model, [op("scores.a.rent", "scale", 1.2)])
    assert r.after.feasibility["a"] == 0.0 and r.after.win_rates["b"] == 1.0
    assert r.winner_changed and r.level_changed


def test_unpatched_inputs_draw_the_same_values(rent_model):
    after, _ = apply_patch(rent_model, [op("scores.a.commute", "scale", 1.5)])
    d0, d1 = draw(rent_model, 1000, 0), draw(after, 1000, 0)
    assert np.array_equal(d0.quantile, d1.quantile)
    assert np.array_equal(d0.raw[:, 1, :], d1.raw[:, 1, :])


@pytest.mark.parametrize(
    "target, kind, value",
    [
        ("scores.z.rent", "set", 1),
        ("scores.a", "set", 1),
        ("nonsense", "set", 1),
        ("constraints.5.threshold", "set", 1),
        ("scores.a.rent", "scale", 0),
        ("scores.a.rent", "scale", -1),
        ("weights.rent", "set", 1.5),
        ("weights.rent", "shift", -0.9),
        ("weights.nope", "set", 0.5),
        ("constraints.x.threshold", "set", 1),
    ],
)
def test_invalid_patches(rent_model, target, kind, value):
    with pytest.raises(PatchError):
        apply_patch(rent_model, [op(target, kind, value)])


def test_collapsing_range_is_a_patch_error(rent_model):
    with pytest.raises(PatchError):
        apply_patch(rent_model, [op("scores.a.commute", "shift", 1e20)])


def test_weight_patch_needs_other_criteria(hand_model):
    with pytest.raises(PatchError):
        apply_patch(hand_model, [op("weights.rent", "scale", 0)])  # scale 0 invalid
    one = hand_model.model_dump()
    one["criteria"] = one["criteria"][:1]
    one["scores"] = {k: {"rent": v["rent"]} for k, v in one["scores"].items()}
    one["weights"]["values"] = {"rent": 1}
    from quanheng.choice.model import ChoiceModel

    with pytest.raises(PatchError):
        apply_patch(ChoiceModel.model_validate(one), [op("weights.rent", "set", 0.5)])


def test_anchors_are_not_a_target(rent_model):
    with pytest.raises(PatchError, match="锚点"):
        apply_patch(rent_model, [op("criteria.rent.bad", "set", 100000)])
