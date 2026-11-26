"""Reading and writing decision files."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from quanheng import zh
from quanheng.choice.model import ChoiceModel
from quanheng.errors import ModelFileError

_VALUE_ERROR_PREFIX = "Value error, "


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [zh.FILE_INVALID.format(path=path)]
    for err in exc.errors():
        msg = err["msg"].removeprefix(_VALUE_ERROR_PREFIX)
        loc = ".".join(str(part) for part in err["loc"])
        lines.append(f"{loc}：{msg}" if loc else msg)
    return "\n".join(lines)


def load_model(path: Path) -> ChoiceModel:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ModelFileError(zh.FILE_NOT_FOUND.format(path=path)) from exc
    except OSError as exc:
        raise ModelFileError(zh.FILE_UNREADABLE.format(path=path, reason=exc)) from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ModelFileError(zh.FILE_NOT_UTF8.format(path=path)) from exc
    if not text.strip():
        raise ModelFileError(zh.FILE_EMPTY.format(path=path))
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelFileError(zh.FILE_BAD_JSON.format(path=path, reason=exc)) from exc
    if not isinstance(data, dict):
        raise ModelFileError(zh.FILE_NOT_OBJECT.format(path=path))
    kind = data.get("kind", "choice")
    if kind != "choice":
        raise ModelFileError(zh.FILE_WRONG_KIND.format(path=path, kind=json.dumps(kind)))
    try:
        return ChoiceModel.model_validate_json(text)
    except ValidationError as exc:
        raise ModelFileError(_format_validation_error(path, exc)) from exc


def save_model(model: ChoiceModel, path: Path) -> None:
    text = json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ModelFileError(zh.FILE_CANNOT_WRITE.format(path=path, reason=exc)) from exc
