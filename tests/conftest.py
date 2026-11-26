import copy
from pathlib import Path
from typing import Any

import pytest

from quanheng.choice.model import ChoiceModel


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="rewrite the files in examples/golden/ from the current engine output",
    )


HAND: dict[str, Any] = {
    "title": "hand",
    "options": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
    "criteria": [
        {"id": "rent", "name": "房租", "unit": "日元/月", "bad": 90000, "good": 50000},
        {"id": "commute", "name": "通勤", "unit": "分钟", "bad": 60, "good": 20},
    ],
    "scores": {"a": {"rent": 70000, "commute": 30}, "b": {"rent": 60000, "commute": 50}},
    "weights": {"values": {"rent": 1, "commute": 1}, "certainty": "fixed"},
}

EXAMPLE_PATH = Path(__file__).parent.parent / "examples" / "rent-a-or-b.json"


@pytest.fixture
def hand_dict() -> dict[str, Any]:
    return copy.deepcopy(HAND)


@pytest.fixture
def hand_model() -> ChoiceModel:
    return ChoiceModel.model_validate(copy.deepcopy(HAND))


@pytest.fixture
def rent_model() -> ChoiceModel:
    return ChoiceModel.model_validate_json(EXAMPLE_PATH.read_text(encoding="utf-8"))
