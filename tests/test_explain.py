import pytest

from quanheng.choice.explain import explain_choice, explain_whatif, fmt_number, fmt_percent
from quanheng.choice.model import ChoiceModel
from quanheng.choice.result import Thresholds, TippingPoint, Verdict
from quanheng.choice.solve import solve_choice
from quanheng.whatif import PatchOp, compare


@pytest.mark.parametrize(
    "x, text",
    [
        (39.99999, "40"),
        (41.1094, "41.11"),
        (83469.5, "83,470"),
        (78000, "78,000"),
        (0.5, "0.5"),
        (-1234.4, "-1,234"),
        (0, "0"),
    ],
)
def test_fmt_number(x, text):
    assert fmt_number(x) == text


@pytest.mark.parametrize(
    "p, text",
    [
        (0, "0%"),
        (0.001, "<1%"),
        (0.0049, "<1%"),
        (0.005, "1%"),
        (0.5, "50%"),
        (0.8092, "81%"),
        (0.9949, "99%"),
        (0.995, ">99%"),
        (0.99946, ">99%"),
        (1, "100%"),
    ],
)
def test_fmt_percent(p, text):
    assert fmt_percent(p) == text


def test_hand_model_text(hand_model):
    lines = explain_choice(hand_model, solve_choice(hand_model)).splitlines()
    assert lines[0] == "hand"
    assert "明确 A　A 在 100% 的情况下胜出（±0.0%）" in lines
    assert "在你给出的范围和偏好之内成立，不代表事后一定更好。" in lines
    assert "为什么是 A（按中心值计算，A 相对 B 的各项贡献）" in lines
    assert "  通勤 +0.25" in lines
    assert '  · B 的"通勤"低于 40 分钟（你填的是 50）' in lines
    assert '  · "房租"的权重高于 0.67（现在是 0.50）' in lines
    assert "  · 其余 1 个输入在参照范围内怎么变都不会翻转结论" in lines
    assert "  没有哪一个输入单独弄准后会明显改变把握。" in lines
    assert "  · 各个不确定的输入相互独立" in lines
    assert lines[-1].startswith("抽样 10,000 次，种子 0，引擎 0.1.0，NumPy ")


def test_example_text(rent_model):
    text = explain_choice(rent_model, solve_choice(rent_model))
    assert "倾向 A　A 在 81% 的情况下胜出（±0.8%）" in text
    assert '  · B 的"通勤"低于 41.11 分钟（你填的是 40–55）' in text
    assert '  1. B 的"通勤"：在你给的范围低端时 A 的胜出率是' in text
    assert "rent" not in text and "commute" not in text  # 不出现内部标识


def test_hidden_flipping_inputs_are_counted(rent_model):
    result = solve_choice(rent_model)
    assert sum(t.flips_at is not None for t in result.tipping_points) == 11
    lines = explain_choice(rent_model, result).splitlines()
    start = lines.index("什么会让结论翻转")
    block = lines[start + 1 : start + 8]
    assert block[5] == "  · 另有 6 个输入也会翻转结论，但临界值离你填的值更远"
    assert block[6] == "  · 其余 1 个输入在参照范围内怎么变都不会翻转结论"


def test_no_hidden_line_when_all_flips_are_shown(hand_model):
    text = explain_choice(hand_model, solve_choice(hand_model))
    assert "另有" not in text


def test_flip_at_the_central_value_has_no_direction():
    m = _single([0, 0, 10], 2)
    lines = explain_choice(m, solve_choice(m)).splitlines()
    assert '  · 如果 A 的"项"确定就是 0（你填的中心值），结论就会翻转' in lines
    assert not any('A 的"项"高于' in line or 'A 的"项"低于' in line for line in lines)


def test_flip_at_the_central_value_with_unit_and_weight(hand_model):
    result = solve_choice(hand_model)
    at_current = [
        TippingPoint(
            target="scores.a.rent",
            current=70000,
            flips_at=70000,
            new_leader="b",
            within_stated_range=True,
        ),
        TippingPoint(
            target="weights.rent",
            current=0.5,
            flips_at=0.5,
            new_leader="b",
            within_stated_range=True,
        ),
    ]
    lines = explain_choice(
        hand_model, result.model_copy(update={"tipping_points": at_current})
    ).splitlines()
    assert '  · 如果 A 的"房租"确定就是 70,000 日元/月（你填的中心值），结论就会翻转' in lines
    assert '  · 如果"房租"的权重确定就是 0.50，结论就会翻转' in lines


def test_verdict_does_not_round_up_onto_the_clear_threshold():
    m = _single(5, [0, 5 / 0.8975])
    r = solve_choice(m)
    assert r.verdict.win_rate == 0.8994 and r.verdict.level == "leaning"
    assert explain_choice(m, r).splitlines()[1] == "倾向 A　A 在 89.9% 的情况下胜出（±0.6%）"


def test_verdict_near_certain_is_not_shown_as_100():
    m = _single(5, [0, 5.002])
    r = solve_choice(m, samples=100_000)
    assert r.verdict.win_rate == 0.99946
    assert explain_choice(m, r).splitlines()[1] == "明确 A　A 在 >99% 的情况下胜出（±0.0%）"


def _with_verdict(result, level, win_rate, thresholds=None):
    verdict = Verdict(level=level, winner="a", runner_up="b", win_rate=win_rate, std_error=0.0)
    update = {"verdict": verdict}
    if thresholds is not None:
        update["meta"] = result.meta.model_copy(update={"thresholds": thresholds})
    return result.model_copy(update=update)


def test_verdict_threshold_rule_uses_the_recorded_thresholds(hand_model):
    r = solve_choice(hand_model)
    line = explain_choice(hand_model, _with_verdict(r, "toss_up", 0.6996)).splitlines()[1]
    assert line == "难分高下　A 略占上风，在 69.9% 的情况下胜出（±0.0%）"
    custom = Thresholds(clear=0.8, leaning=0.6)
    line = explain_choice(hand_model, _with_verdict(r, "leaning", 0.7996, custom)).splitlines()[1]
    assert line == "倾向 A　A 在 79.9% 的情况下胜出（±0.0%）"
    line = explain_choice(hand_model, _with_verdict(r, "leaning", 0.8996)).splitlines()[1]
    assert line == "倾向 A　A 在 89.9% 的情况下胜出（±0.0%）"
    line = explain_choice(hand_model, _with_verdict(r, "clear", 0.8996, custom)).splitlines()[1]
    assert line == "明确 A　A 在 90% 的情况下胜出（±0.0%）"  # 门槛 0.8，不矛盾


def test_tiny_infeasible_share_is_not_shown_as_zero():
    m = ChoiceModel.model_validate(
        {
            "title": "t",
            "options": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            "criteria": [{"id": "x", "name": "项", "bad": 0, "good": 10}],
            "constraints": [{"criterion": "x", "op": ">=", "threshold": 0.01}],
            "scores": {"a": {"x": [0, 10]}, "b": {"x": [0, 10]}},
            "weights": {"values": {"x": 1}, "certainty": "fixed"},
        }
    )
    r = solve_choice(m)
    assert 0 < 1 - r.feasibility["a"] < 0.005
    lines = explain_choice(m, r).splitlines()
    assert "A 有 <1% 的情况不满足硬约束。" in lines
    assert not any("0% 的情况不满足" in line for line in lines)


def test_mostly_infeasible_text(hand_model):
    data = hand_model.model_dump()
    data["constraints"] = [{"criterion": "rent", "op": "<=", "threshold": 50000}]
    m = ChoiceModel.model_validate(data)
    text = explain_choice(m, solve_choice(m))
    assert "多数情况下没有可行方案　100% 的情况下没有任何候选满足全部硬约束" in text
    assert '  · "房租" <= 50,000：最接近的是 B，还差 10,000 日元/月' in text
    assert "什么会让结论翻转" not in text and "为什么是" not in text


def test_constraint_met_at_central_values_but_not_in_most_samples():
    m = ChoiceModel.model_validate(
        {
            "title": "t",
            "options": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            "criteria": [{"id": "x", "name": "项", "bad": 0, "good": 10}],
            "constraints": [{"criterion": "x", "op": ">=", "threshold": 5}],
            "scores": {"a": {"x": [0, 5, 5.5]}, "b": {"x": 3}},  # A 最可能是 5，但多数抽样低于 5
            "weights": {"values": {"x": 1}, "certainty": "fixed"},
        }
    )
    r = solve_choice(m)
    assert r.verdict.level == "mostly_infeasible"
    assert (r.constraint_gaps[0].nearest_option, r.constraint_gaps[0].gap) == ("a", 0.0)
    lines = explain_choice(m, r).splitlines()
    assert '  · "项" >= 5：按中心值 A 能满足，但抽样中多数情况下不满足' in lines
    assert not any("还差" in line for line in lines)


def test_whatif_text(hand_model):
    r = compare(hand_model, [PatchOp(target="weights.rent", op="set", value=0.8)])
    lines = explain_whatif(hand_model, r).splitlines()
    assert lines[0] == '"房租"的权重：0.50 → 0.80'
    assert lines[1] == "之前：明确 A　A 在 100% 的情况下胜出（±0.0%）"
    assert lines[2] == "之后：明确 B　B 在 100% 的情况下胜出（±0.0%）"
    assert lines[-1] == "结论翻转了。"


def _model(hand_dict, edit):
    edit(hand_dict)
    return ChoiceModel.model_validate(hand_dict)


def _single(a, b):
    return ChoiceModel.model_validate(
        {
            "title": "t",
            "options": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            "criteria": [{"id": "x", "name": "项", "bad": 0, "good": 10}],
            "scores": {"a": {"x": a}, "b": {"x": b}},
            "weights": {"values": {"x": 1}, "certainty": "fixed"},
        }
    )


def test_toss_up_text():
    m = _single(5, [0, 10])
    lines = explain_choice(m, solve_choice(m)).splitlines()
    assert lines[1].startswith("难分高下　")
    assert "略占上风" in lines[1]


def test_central_disagrees_text():
    m = _single([0, 0, 10], 2)
    r = solve_choice(m)
    assert r.verdict.winner == "a" and r.central_leader == "b"
    lines = explain_choice(m, r).splitlines()
    assert "为什么是 A（按中心值计算，A 相对 B 的各项贡献）" in lines
    assert "注意：按中心值计算是 B 领先，与抽样结果不一致，以抽样结果为准。" in lines


def test_outside_anchors_text(hand_dict):
    m = _model(hand_dict, lambda d: d["scores"]["a"].update(rent=95000))
    lines = explain_choice(m, solve_choice(m)).splitlines()
    assert '注意：A 的"房租"超出了你设的参照范围。' in lines


def test_poor_consistency_text(hand_dict):
    m = _model(hand_dict, lambda d: d["weights"].update(consistency_ratio=0.2))
    lines = explain_choice(m, solve_choice(m)).splitlines()
    assert "注意：两两比较的结果前后不够一致，权重的波动已放大。" in lines


def test_pinning_into_nothing_feasible_is_listed_as_a_flip(hand_dict):
    def edit(d):
        d["criteria"][0]["good"] = 60000
        d["scores"]["a"].update(rent=78000, commute=[30, 40])
        d["scores"]["b"].update(rent=88000, commute=[20, 30])
        d["constraints"] = [{"criterion": "rent", "op": "<=", "threshold": 85000}]

    m = _model(hand_dict, edit)
    lines = explain_choice(m, solve_choice(m)).splitlines()
    assert '  · A 的"房租"高于 85,000 日元/月（你填的是 78,000）' in lines
    # A 的房租、B 的房租会翻转；A 的通勤、B 的通勤、两条权重不会
    assert "  · 其余 4 个输入在参照范围内怎么变都不会翻转结论" in lines


def test_partial_infeasibility_text(hand_dict):
    def edit(d):
        d["scores"]["a"]["rent"] = [70000, 90000]
        d["constraints"] = [{"criterion": "rent", "op": "<=", "threshold": 85000}]

    m = _model(hand_dict, edit)
    text = explain_choice(m, solve_choice(m))
    assert "A 有 25% 的情况不满足硬约束。" in text.splitlines()
    assert "没有任何候选可行" not in text
