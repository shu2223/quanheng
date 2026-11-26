import copy
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from quanheng.choice.model import ChoiceModel


def test_example_and_hand_model_load(rent_model: ChoiceModel, hand_model: ChoiceModel) -> None:
    assert rent_model.title == "租 A 还是租 B？" and len(rent_model.criteria) == 4
    assert hand_model.weights.normalized() == {"rent": 0.5, "commute": 0.5}
    assert rent_model.unconfirmed_inputs() == [] == hand_model.unconfirmed_inputs()


def test_unconfirmed_inputs_are_listed(rent_model: ChoiceModel) -> None:
    data = rent_model.model_dump()
    data["scores"]["b"]["commute"]["confirmed"] = False
    assert ChoiceModel.model_validate(data).unconfirmed_inputs() == ["scores.b.commute"]


def mutate(d: dict[str, Any], fn: Callable[[dict[str, Any]], object]) -> dict[str, Any]:
    d = copy.deepcopy(d)
    fn(d)
    return d


@pytest.mark.parametrize(
    "fn",
    [
        lambda d: d["options"].pop(),  # only one option left
        lambda d: d.update(criteria=[]),  # no criteria
        lambda d: d["options"].append({"id": "a", "name": "重复"}),  # duplicate option id
        lambda d: d["options"][0].update(id="a.b"),  # illegal id characters
        lambda d: d["criteria"][0].update(good=90000),  # bad == good
        lambda d: d["scores"]["b"].pop("commute"),  # missing value
        lambda d: d["scores"].update(c={"rent": 1, "commute": 1}),  # extra option scores
        lambda d: d.update(constraints=[{"criterion": "nope", "op": "<=", "threshold": 1}]),
        lambda d: d["weights"]["values"].pop("commute"),  # missing weight
        lambda d: d.update(schema_version=2),
    ],
)
def test_invalid_models_rejected(hand_dict: dict[str, Any], fn: Callable[..., object]) -> None:
    with pytest.raises(ValidationError):
        ChoiceModel.model_validate(mutate(hand_dict, fn))


@pytest.mark.parametrize(
    ("fn", "message"),
    [
        (lambda d: d["scores"]["b"].pop("commute"), "scores.b 缺少准则 commute"),
        (lambda d: d["scores"].update(c={"rent": 1, "commute": 1}), "scores 多出候选 c"),
        (
            lambda d: d.update(constraints=[{"criterion": "nope", "op": "<=", "threshold": 1}]),
            "constraints.0 引用了不存在的准则 nope",
        ),
        (lambda d: d["weights"]["values"].pop("commute"), "weights.values 缺少准则 commute"),
    ],
)
def test_cross_field_errors_name_the_place(
    hand_dict: dict[str, Any], fn: Callable[..., object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message.replace(".", r"\.")):
        ChoiceModel.model_validate(mutate(hand_dict, fn))
