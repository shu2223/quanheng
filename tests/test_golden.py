import json
from pathlib import Path
from typing import Any

from quanheng.choice.explain import explain_choice
from quanheng.choice.model import ChoiceModel
from quanheng.choice.solve import solve_choice
from quanheng.io import load_model, save_model

GOLDEN = Path(__file__).parent.parent / "examples" / "golden" / "rent-a-or-b.result.json"
GOLDEN_TEXT = GOLDEN.parent / "rent-a-or-b.explain.txt"


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
    view["meta"].pop("scipy_version", None)
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


def explanation_without_meta(model: ChoiceModel) -> str:
    """The explanation minus its last line (versions differ by platform), ending in one LF."""
    lines = explain_choice(model, solve_choice(model)).split("\n")
    assert lines[-1].startswith("抽样 ") and lines[-2] == ""
    return "\n".join(lines[:-2]) + "\n"


def test_example_explanation_matches_golden(rent_model, request):
    # 跨操作系统：显示的数字和结论措辞必须一致，所以逐字比较
    text = explanation_without_meta(rent_model)
    if request.config.getoption("--update-golden"):
        GOLDEN_TEXT.write_bytes(text.encode("utf-8"))  # LF on every platform
    assert GOLDEN_TEXT.read_bytes().decode("utf-8") == text


def test_save_and_load_round_trip(rent_model, tmp_path):
    save_model(rent_model, tmp_path / "m.json")
    assert load_model(tmp_path / "m.json") == rent_model
    assert "租 A 还是租 B？" in (tmp_path / "m.json").read_text(encoding="utf-8")
