from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    """Base class of every Quanheng model: no extra fields, no NaN or infinity."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
