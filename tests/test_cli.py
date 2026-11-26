import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
EXAMPLE = "examples/rent-a-or-b.json"


def run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "quanheng", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )


def test_solve_prints_the_explanation():
    p = run("solve", EXAMPLE)
    assert p.returncode == 0 and p.stdout.splitlines()[0] == "租 A 还是租 B？"
    assert "倾向 A　A 在 81% 的情况下胜出（±0.8%）" in p.stdout and p.stderr == ""


def test_solve_json_and_reproducibility():
    a, b = run("solve", EXAMPLE, "--json"), run("solve", EXAMPLE, "--json")
    data = json.loads(a.stdout)
    assert data["kind"] == "choice_result" and data["meta"]["seed"] == 0
    assert a.stdout == b.stdout
    other = json.loads(run("solve", EXAMPLE, "--json", "--seed", "1").stdout)
    assert other["win_rates"] != data["win_rates"]


def test_whatif_applies_ops_in_command_line_order():
    p = run("whatif", EXAMPLE, "--scale", "scores.a.rent=1.2")
    assert p.returncode == 0 and p.stdout.splitlines()[-1] == "结论翻转了。"
    p = run(
        "whatif", EXAMPLE, "--scale", "scores.a.rent=2", "--set", "scores.a.rent=78000", "--json"
    )
    assert json.loads(p.stdout)["changes"][-1]["after"] == [78000.0]


def test_version():
    assert run("--version").stdout.strip() == "quanheng 0.1.0"


def test_console_that_cannot_show_chinese():
    p = run("solve", EXAMPLE, env={**os.environ, "PYTHONIOENCODING": "ascii"})
    assert p.returncode == 0 and "倾向 A" in p.stdout


def test_file_with_bom_loads(tmp_path):
    f = tmp_path / "bom.json"
    f.write_text((ROOT / EXAMPLE).read_text(encoding="utf-8"), encoding="utf-8-sig")
    assert run("solve", str(f)).returncode == 0


@pytest.mark.parametrize("content", ["", "{not json", "[]", '{"kind": "timing"}'])
def test_broken_files(tmp_path, content):
    f = tmp_path / "broken.json"
    f.write_text(content, encoding="utf-8")
    p = run("solve", str(f))
    assert p.returncode == 2 and p.stderr.strip() and "Traceback" not in p.stderr


def test_missing_file_and_bad_arguments():
    for args in (
        ["solve", "no-such-file.json"],
        ["solve", EXAMPLE, "--samples", "50"],
        ["solve", EXAMPLE, "--samples", "10000000"],
        ["whatif", EXAMPLE, "--set", "criteria.rent.bad=1"],
        ["whatif", EXAMPLE, "--set", "scores.a.rent"],
        ["whatif", EXAMPLE, "--set", "scores.a.rent=abc"],
    ):
        p = run(*args)
        assert p.returncode == 2 and "Traceback" not in p.stderr, args


@pytest.mark.parametrize(
    ("fn", "place"),
    [
        (lambda d: d["scores"]["a"].update(rent=[5, 5]), "scores.a.rent"),  # 字段本身不合法
        (lambda d: d["scores"]["b"].pop("size"), "scores.b"),  # 跨字段：缺一个数值
    ],
)
def test_validation_errors_name_the_place(tmp_path, fn, place):
    data = json.loads((ROOT / EXAMPLE).read_text(encoding="utf-8"))
    fn(data)
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    p = run("solve", str(f))
    assert p.returncode == 2 and place in p.stderr and "Traceback" not in p.stderr


def test_nan_is_rejected(tmp_path):
    text = (ROOT / EXAMPLE).read_text(encoding="utf-8").replace('"rent": 78000', '"rent": NaN')
    f = tmp_path / "nan.json"
    f.write_text(text, encoding="utf-8")
    assert run("solve", str(f)).returncode == 2


def test_unconfirmed_inputs_exit_3(tmp_path):
    text = (
        (ROOT / EXAMPLE)
        .read_text(encoding="utf-8")
        .replace('"confirmed": true', '"confirmed": false')
    )
    f = tmp_path / "u.json"
    f.write_text(text, encoding="utf-8")
    p = run("solve", str(f))
    assert p.returncode == 3 and "scores.b.commute" in p.stderr
