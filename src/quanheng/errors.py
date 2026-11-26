class QuanhengError(Exception):
    """Base class of all errors raised by the Quanheng engine."""


class UnconfirmedInputError(QuanhengError):
    """Raised when a quoted or LLM-inferred value has not been confirmed."""

    def __init__(self, targets: list[str]) -> None:
        self.targets = targets
        super().__init__("unconfirmed inputs: " + ", ".join(targets))


class PatchError(QuanhengError):
    """Raised when a patch cannot be applied to a model."""


class ModelFileError(QuanhengError):
    """Raised when a model file cannot be read or is invalid."""
