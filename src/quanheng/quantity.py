from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator

from quanheng._base import Model


class Exact(Model):
    kind: Literal["exact"] = "exact"
    value: float


class Range(Model):
    kind: Literal["range"] = "range"
    low: float
    high: float

    @model_validator(mode="after")
    def _check_order(self) -> "Range":
        if not self.low < self.high:
            raise ValueError("range requires low < high")
        return self


class ThreePoint(Model):
    kind: Literal["three_point"] = "three_point"
    low: float
    mode: float
    high: float

    @model_validator(mode="after")
    def _check_order(self) -> "ThreePoint":
        if not (self.low <= self.mode <= self.high and self.low < self.high):
            raise ValueError("three-point estimate requires low <= mode <= high and low < high")
        return self


class UserInput(Model):
    type: Literal["user_input"] = "user_input"


class Quoted(Model):
    type: Literal["quoted"] = "quoted"
    quote: Annotated[str, StringConstraints(min_length=1)]


class Imported(Model):
    type: Literal["imported"] = "imported"
    file: str
    location: str


class LlmInferred(Model):
    type: Literal["llm_inferred"] = "llm_inferred"
    reason: str


def _is_number(x: object) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


class Quantity(Model):
    value: Annotated[Exact | Range | ThreePoint, Field(discriminator="kind")]
    unit: str = ""
    source: Annotated[UserInput | Quoted | Imported | LlmInferred, Field(discriminator="type")] = (
        UserInput()
    )
    confirmed: bool = False

    @model_validator(mode="before")
    @classmethod
    def _expand_shorthand(cls, data: Any) -> Any:
        if _is_number(data):
            return {"value": {"kind": "exact", "value": data}}
        if isinstance(data, list):
            if not all(_is_number(x) for x in data):
                raise ValueError("shorthand lists may contain only numbers")
            if len(data) == 2:
                return {"value": {"kind": "range", "low": data[0], "high": data[1]}}
            if len(data) == 3:
                return {
                    "value": {
                        "kind": "three_point",
                        "low": data[0],
                        "mode": data[1],
                        "high": data[2],
                    }
                }
            raise ValueError("shorthand list must have 2 (range) or 3 (three-point) numbers")
        return data

    @property
    def needs_confirmation(self) -> bool:
        return self.source.type in ("quoted", "llm_inferred") and not self.confirmed

    @property
    def central(self) -> float:
        v = self.value
        if isinstance(v, Exact):
            return v.value
        if isinstance(v, Range):
            return (v.low + v.high) / 2
        return v.mode

    @property
    def bounds(self) -> tuple[float, float]:
        v = self.value
        if isinstance(v, Exact):
            return (v.value, v.value)
        return (v.low, v.high)

    @property
    def is_uncertain(self) -> bool:
        return not isinstance(self.value, Exact)
