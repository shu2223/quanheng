import ast
from pathlib import Path

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


def imported_names(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def test_engine_does_not_import_outer_layers_or_network() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.relative_to(SRC).parts[0] in OUTER_LAYERS:
            continue
        for name in imported_names(path):
            if name.split(".")[0] in FORBIDDEN_TOP or name.startswith(FORBIDDEN_PREFIX):
                offenders.append(f"{path.relative_to(SRC)}: {name}")
    assert offenders == []
