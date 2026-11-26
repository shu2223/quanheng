import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src" / "quanheng"
OUTER_LAYERS = {"llm", "server"}
FORBIDDEN_TOP = {
    "openai",
    "anthropic",
    "httpx",
    "requests",
    "aiohttp",
    "urllib3",
    "fastapi",
    "starlette",
    "uvicorn",
    "socket",
}
FORBIDDEN_PREFIX = ("quanheng.llm", "quanheng.server")


def package_of(path: Path) -> str:
    """The package a module under src/quanheng belongs to, e.g. ``quanheng.choice``."""
    return ".".join(["quanheng", *path.relative_to(SRC).parts[:-1]])


def imported_names(source: str, package: str) -> list[str]:
    """Absolute names the source imports, relative imports resolved against `package`.

    ``from X import y`` yields both ``X`` and ``X.y``, because ``y`` may be a submodule
    (``from quanheng import llm``, ``from . import server``).
    """
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level > 0:
                parts = package.split(".")
                anchor = parts[: max(len(parts) - (node.level - 1), 0)]
                base = ".".join([*anchor, *([node.module] if node.module else [])])
            names.append(base)
            names += [f"{base}.{a.name}" for a in node.names]
    return names


def is_forbidden(name: str) -> bool:
    if name.split(".")[0] in FORBIDDEN_TOP:
        return True
    return any(name == p or name.startswith(p + ".") for p in FORBIDDEN_PREFIX)


def offending_imports(source: str, package: str) -> list[str]:
    return [name for name in imported_names(source, package) if is_forbidden(name)]


def test_engine_does_not_import_outer_layers_or_network() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.relative_to(SRC).parts[0] in OUTER_LAYERS:
            continue
        source = path.read_text(encoding="utf-8")
        for name in offending_imports(source, package_of(path)):
            offenders.append(f"{path.relative_to(SRC)}: {name}")
    assert offenders == []


@pytest.mark.parametrize(
    ("source", "package"),
    [
        ("import quanheng.llm", "quanheng"),
        ("from quanheng.server.app import run", "quanheng.choice"),
        ("from quanheng import llm", "quanheng"),
        ("from quanheng import zh, server as s", "quanheng.choice"),
        ("from .llm import draft", "quanheng"),
        ("from . import server", "quanheng"),
        ("from ..llm import draft", "quanheng.choice"),
        ("from .. import server", "quanheng.choice"),
        ("from ..server.app import run", "quanheng.choice"),
        ("def f():\n    from .llm import draft", "quanheng"),
        ("import requests", "quanheng"),
        ("from socket import create_connection", "quanheng.choice"),
    ],
)
def test_checker_flags_outer_layers_and_network(source: str, package: str) -> None:
    assert offending_imports(source, package) != []


@pytest.mark.parametrize(
    ("source", "package"),
    [
        ("from quanheng import zh", "quanheng.choice"),
        ("from . import zh", "quanheng"),
        ("from .model import ChoiceModel", "quanheng.choice"),
        ("from ..weights import noise_factors", "quanheng.choice"),
        ("from quanheng.choice.model import ChoiceModel", "quanheng"),
        ("import numpy as np", "quanheng"),
    ],
)
def test_checker_allows_engine_imports(source: str, package: str) -> None:
    assert offending_imports(source, package) == []


def test_package_of() -> None:
    assert package_of(SRC / "cli.py") == "quanheng"
    assert package_of(SRC / "choice" / "explain.py") == "quanheng.choice"
    assert package_of(SRC / "choice" / "__init__.py") == "quanheng.choice"
