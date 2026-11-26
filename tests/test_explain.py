import pytest

from quanheng.choice.explain import explain_choice, explain_whatif, fmt_number
from quanheng.choice.model import ChoiceModel
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


def test_mostly_infeasible_text(hand_model):
    data = hand_model.model_dump()
    data["constraints"] = [{"criterion": "rent", "op": "<=", "threshold": 50000}]
    m = ChoiceModel.model_validate(data)
    text = explain_choice(m, solve_choice(m))
    assert "多数情况下没有可行方案　100% 的情况下没有任何候选满足全部硬约束" in text
    assert '  · "房租" <= 50,000：最接近的是 B，还差 10,000 日元/月' in text
    assert "什么会让结论翻转" not in text and "为什么是" not in text


def test_whatif_text(hand_model):
    r = compare(hand_model, [PatchOp(target="weights.rent", op="set", value=0.8)])
    lines = explain_whatif(hand_model, r).splitlines()
    assert lines[0] == '"房租"的权重：0.50 → 0.80'
    assert lines[1] == "之前：明确 A　A 在 100% 的情况下胜出（±0.0%）"
    assert lines[2] == "之后：明确 B　B 在 100% 的情况下胜出（±0.0%）"
    assert lines[-1] == "结论翻转了。"
