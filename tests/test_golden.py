import json
from pathlib import Path
from typing import Any

from quanheng.choice.solve import solve_choice
from quanheng.io import load_model, save_model

GOLDEN = Path(__file__).parent.parent / "examples" / "golden" / "rent-a-or-b.result.json"


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def rounded_view(data: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = _round(data)
    view["meta"].pop("engine_version", None)
    view["meta"].pop("numpy_version", None)
    return view


def test_example_matches_golden(rent_model, request):
    view = rounded_view(solve_choice(rent_model).model_dump())
    if request.config.getoption("--update-golden"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(view, ensure_ascii=False, indent=2) + "\n"
        GOLDEN.write_bytes(text.encode("utf-8"))  # LF on every platform
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert view == golden
    assert golden["verdict"]["level"] == "leaning"
    assert golden["win_rates"] == {"a": 0.8092, "b": 0.1908}


def test_save_and_load_round_trip(rent_model, tmp_path):
    save_model(rent_model, tmp_path / "m.json")
    assert load_model(tmp_path / "m.json") == rent_model
    assert "租 A 还是租 B？" in (tmp_path / "m.json").read_text(encoding="utf-8")
