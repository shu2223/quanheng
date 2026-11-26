import pytest
from pydantic import ValidationError

from quanheng.errors import ModelFileError, PatchError, QuanhengError, UnconfirmedInputError
from quanheng.quantity import Exact, Quantity


def q(**kw):
    return Quantity.model_validate(kw)


def test_shorthand_number():
    x = Quantity.model_validate(70000)
    assert x.value == Exact(value=70000) and x.source.type == "user_input"
    assert x.central == 70000 and x.bounds == (70000, 70000) and not x.is_uncertain


def test_shorthand_range_and_three_point():
    assert Quantity.model_validate([25, 35]).central == 30
    x = Quantity.model_validate([4, 5, 7])
    assert x.central == 5 and x.bounds == (4, 7) and x.is_uncertain


@pytest.mark.parametrize(
    "source, confirmed, expected",
    [
        ({"type": "user_input"}, False, False),
        ({"type": "imported", "file": "a.csv", "location": "B2"}, False, False),
        ({"type": "quoted", "quote": "房租7万"}, False, True),
        ({"type": "quoted", "quote": "房租7万"}, True, False),
        ({"type": "llm_inferred", "reason": "估算"}, False, True),
        ({"type": "llm_inferred", "reason": "估算"}, True, False),
    ],
)
def test_confirmation_rule(source, confirmed, expected):
    x = q(value={"kind": "exact", "value": 1}, source=source, confirmed=confirmed)
    assert x.needs_confirmation is expected


@pytest.mark.parametrize(
    "bad",
    [
        [5, 5],
        [7, 3],
        [1, 0, 2],
        [0, 3, 2],
        [0, 0, 0],
        [1, 2, 3, 4],
        [],
        [1],
        [1, "2"],
        [1, True],
        float("nan"),
        float("inf"),
        "70000",
        True,
        None,
    ],
)
def test_rejects_invalid_values(bad):
    with pytest.raises(ValidationError):
        Quantity.model_validate(bad)


def test_quoted_requires_quote_and_extra_fields_rejected():
    with pytest.raises(ValidationError):
        q(value={"kind": "exact", "value": 1}, source={"type": "quoted", "quote": ""})
    with pytest.raises(ValidationError):
        q(value={"kind": "exact", "value": 1}, typo=1)


VALUES = [
    {"kind": "exact", "value": 3},
    {"kind": "range", "low": 40, "high": 55},
    {"kind": "three_point", "low": 1, "mode": 2, "high": 4},
]
SOURCES = [
    {"type": "user_input"},
    {"type": "quoted", "quote": "房租7万"},
    {"type": "imported", "file": "a.csv", "location": "B2"},
    {"type": "llm_inferred", "reason": "按地图估算"},
]


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("source", SOURCES)
def test_round_trip(value, source):
    x = q(value=value, unit="分钟", source=source, confirmed=True)
    assert Quantity.model_validate_json(x.model_dump_json()) == x
    assert Quantity.model_validate(x.model_dump()) == x


def test_full_form_dict_still_validates():
    x = q(value={"kind": "range", "low": 40, "high": 55}, source={"type": "user_input"})
    assert x.bounds == (40, 55)


@pytest.mark.parametrize(
    "value",
    [
        {"kind": "exact", "value": "70000"},
        {"kind": "exact", "value": True},
        {"kind": "range", "low": "1", "high": 2},
        {"kind": "three_point", "low": 1, "mode": True, "high": 3},
    ],
)
def test_no_numeric_coercion_in_full_form(value):
    with pytest.raises(ValidationError):
        q(value=value)


@pytest.mark.parametrize("confirmed", ["yes", 1, "true", None])
def test_confirmed_must_be_real_bool(confirmed):
    with pytest.raises(ValidationError):
        q(
            value={"kind": "exact", "value": 1},
            source={"type": "quoted", "quote": "x"},
            confirmed=confirmed,
        )


def test_errors():
    err = UnconfirmedInputError(["scores.b.commute"])
    assert err.targets == ["scores.b.commute"] and "scores.b.commute" in str(err)
    assert isinstance(err, QuanhengError)
    assert issubclass(PatchError, QuanhengError) and issubclass(ModelFileError, QuanhengError)
