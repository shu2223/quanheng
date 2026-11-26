# 权衡引擎与命令行（里程碑 0 + 1a）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建好仓库骨架，并做出不依赖 LLM、不依赖网络的选择模型引擎和命令行，能对一个手写的决策文件求解并输出带稳健度的结论。

**Architecture:** 一个 Python 包 `quanheng`。数值与来源、抽样、权重是三块公共基础；`quanheng.choice` 在其上实现选择模型的抽样、逐次评估、结果汇总、翻盘点和信息价值；`whatif` 做参数修改与前后对比；`explain` 用固定的中文模板把结果写成文字；`cli` 是最外层。所有随机性来自"种子 + 输入标识"决定的独立随机流。

**Tech Stack:** Python ≥ 3.11、NumPy、SciPy（只用 `scipy.special.ndtri`）、Pydantic v2、uv、hatchling、pytest、ruff、mypy、GitHub Actions。

**Spec:** `docs/superpowers/specs/2026-11-26-quanheng-design.md`（执行者两份都要读）

**本计划之外：** 服务层与界面（1b）、LLM 层（1c）各有独立计划，在本计划完成后再写。

## Global Constraints

- Python ≥ 3.11；引擎依赖只有 NumPy、SciPy、Pydantic v2。
- 新增依赖必须是宽松许可证（MIT、BSD、Apache）。因此不使用 Hypothesis（MPL-2.0），性质测试用固定种子的循环写。
- 引擎不导入 LLM 层、服务层和任何网络库，这条规则有自动测试。
- 默认抽样 10,000 次；默认种子 0；允许的抽样次数是 100 到 1,000,000。
- 随机种子、抽样次数、引擎版本、NumPy 版本都写进结果。
- 同一平台、同一引擎版本、同一份锁定依赖：结果逐字节相同。跨操作系统：显示的数字和结论措辞一致。
- 范围按均匀分布抽样，三点估计按三角分布抽样，输入之间相互独立。
- 只要有一个未确认的 `quoted` 或 `llm_inferred` 数值，引擎拒绝计算。
- 结论措辞：最高胜出率 ≥ 90% 为"明确"，70%–90% 为"倾向"，低于 70% 为"难分高下"，"没有任何候选可行"占比最大时为"多数情况下没有可行方案"。
- 面向使用者的文字是中文，全部放在 `src/quanheng/zh.py`。标识符（候选、准则的 `id`）只允许 `[A-Za-z0-9_-]`。
- 所有 Pydantic 模型禁止多余字段，禁止 NaN 和无穷大。
- 直接提交到 `main`。每个任务结束时提交一次。

## Review Focus

设计文档没有逐条写出、但使用者很可能撞上的五类情况。每条的测试已加到对应任务里。

1. **终端编码显示不了中文**（例如日文或英文 Windows 的默认代码页）：命令行必须正常输出、退出码为 0，而不是抛编码错误。→ 任务 12
2. **决策文件带 BOM、是空文件、不是合法 JSON、缺字段**：给出一句能看懂的错误和出错位置，退出码 2，不出现 Python 调用栈。→ 任务 12
3. **数值本身不合法**：范围上下限相等或颠倒、三点估计顺序不对、NaN、无穷大、锚点"差"和"好"相等。在加载时就拒绝，并指明位置。→ 任务 2、5、12
4. **只有一条准则，或某条权重是 0 或 1**：翻盘点计算不能除以零，应当跳过无意义的目标。→ 任务 8
5. **抽样次数过小或过大**（0、负数、50、一千万）：直接拒绝并说明允许范围，而不是算出 NaN 或耗尽内存。→ 任务 6、12

## File Structure

| 文件 | 职责 |
|---|---|
| `pyproject.toml` | 包元数据、依赖、工具配置 |
| `.github/workflows/ci.yml` | 三个操作系统上的检查 |
| `README.md`、`README.zh-CN.md` | 英文、中文说明 |
| `src/quanheng/__init__.py` | `__version__` |
| `src/quanheng/_base.py` | 所有模型的公共基类 |
| `src/quanheng/errors.py` | 异常类型 |
| `src/quanheng/quantity.py` | 带来源的不确定数值 |
| `src/quanheng/sampling.py` | 独立随机流、分位到数值的换算 |
| `src/quanheng/weights.py` | 权重、两两比较、锚点换算、权重波动 |
| `src/quanheng/choice/model.py` | 选择模型的结构与校验 |
| `src/quanheng/choice/evaluate.py` | 抽样与逐次评估 |
| `src/quanheng/choice/result.py` | 结果的结构 |
| `src/quanheng/choice/solve.py` | 汇总成结果 |
| `src/quanheng/choice/tipping.py` | 翻盘点 |
| `src/quanheng/choice/info_value.py` | 最值得补的信息 |
| `src/quanheng/choice/explain.py` | 模板讲解 |
| `src/quanheng/whatif.py` | 参数修改与前后对比 |
| `src/quanheng/zh.py` | 全部中文文案 |
| `src/quanheng/io.py` | 决策文件读写 |
| `src/quanheng/cli.py`、`__main__.py` | 命令行 |
| `examples/rent-a-or-b.json` | 示例决策 |
| `examples/golden/rent-a-or-b.result.json` | 示例的基准结果 |
| `tests/` | 每个模块一个测试文件，`conftest.py` 放公共夹具 |

---

### Task 1: 仓库骨架

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.gitattributes`, `src/quanheng/__init__.py`, `src/quanheng/py.typed`, `.github/workflows/ci.yml`, `README.zh-CN.md`
- Modify: `README.md`
- Test: `tests/test_package.py`, `tests/test_import_rule.py`

**Interfaces:**
- Produces: 可安装的包 `quanheng`，`quanheng.__version__ == "0.1.0"`；命令 `uv run pytest`、`uv run ruff check .`、`uv run ruff format --check .`、`uv run mypy` 全部可用。

- [ ] **Step 1: 写失败的测试**

`tests/test_package.py`：

```python
import quanheng

def test_version():
    assert quanheng.__version__ == "0.1.0"
```

`tests/test_import_rule.py`（设计文档第 6 节的依赖方向检查）：

```python
import ast
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "quanheng"
OUTER_LAYERS = {"llm", "server"}
FORBIDDEN_TOP = {"openai", "anthropic", "httpx", "requests", "aiohttp", "urllib3",
                 "fastapi", "starlette", "uvicorn", "socket"}
FORBIDDEN_PREFIX = ("quanheng.llm", "quanheng.server")

def imported_names(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names

def test_engine_does_not_import_outer_layers_or_network():
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.relative_to(SRC).parts[0] in OUTER_LAYERS:
            continue
        for name in imported_names(path):
            if name.split(".")[0] in FORBIDDEN_TOP or name.startswith(FORBIDDEN_PREFIX):
                offenders.append(f"{path.relative_to(SRC)}: {name}")
    assert offenders == []
```

- [ ] **Step 2: 运行，确认失败**

Run: `uv run pytest`
Expected: 失败，因为还没有 `pyproject.toml` 和包。

- [ ] **Step 3: 写 `pyproject.toml` 和包**

`pyproject.toml` 中必须固定的内容：

```toml
[project]
name = "quanheng"
version = "0.1.0"
description = "A formula-first decision engine: the LLM translates, deterministic math decides."
readme = "README.md"
license = "MIT"
license-files = ["LICENSE"]
requires-python = ">=3.11"
dependencies = ["numpy>=2.0", "scipy>=1.13", "pydantic>=2.7"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6", "mypy>=1.11"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]
ignore = ["RUF001", "RUF002", "RUF003"]  # 中文全角标点会被误报为"易混淆字符"

[tool.mypy]
files = ["src"]
python_version = "3.11"
strict = true
plugins = ["pydantic.mypy"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- `src/quanheng/__init__.py`：用 `importlib.metadata.version("quanheng")` 得到 `__version__`。
- `src/quanheng/py.typed`：空文件。
- `.gitattributes`：`* text=auto eol=lf`。
- `.gitignore`：Python 常规项，加上 `.venv/`、`.mypy_cache/`、`.ruff_cache/`、`.pytest_cache/`、`dist/`。
- 运行 `uv sync`，把生成的 `uv.lock` 一并提交。

- [ ] **Step 4: 写持续集成和 README**

`.github/workflows/ci.yml`：
- 触发：`push` 和 `pull_request`。
- 矩阵：`os: [ubuntu-latest, windows-latest, macos-latest]`，`python: ["3.11", "3.13"]`，`fail-fast: false`。
- 步骤：检出代码；安装 uv（用 `astral-sh/setup-uv` 的当前主版本，查它的 README 确认）；`uv sync --locked --python ${{ matrix.python }}`；`uv run ruff check .`；`uv run ruff format --check .`；`uv run mypy`；`uv run pytest`。

`README.md`（英文）和 `README.zh-CN.md`（中文）内容一一对应，顶部互相链接。各包含这些小节：
1. 一段话说明这是什么，以及原则"LLM 只做翻译，数字和结论全部由确定性代码产生"。
2. 状态：早期开发中；列出设计文档第 12 节的里程碑表，标出当前进度。
3. 开发：`uv sync`、`uv run pytest`。
4. 设计文档链接。
5. 许可证：MIT。

- [ ] **Step 5: 运行全部检查**

Run: `uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: 2 个测试通过，其余无报错。

- [ ] **Step 6: 提交并推送，确认持续集成通过**

```bash
git add -A
git commit -m "chore: project skeleton, CI and bilingual README"
git push origin main
gh run watch --exit-status
```

Expected: 六个矩阵任务全部成功。

---

### Task 2: 带来源的不确定数值

**Files:**
- Create: `src/quanheng/_base.py`, `src/quanheng/errors.py`, `src/quanheng/quantity.py`
- Test: `tests/test_quantity.py`

**Interfaces:**
- Produces:

```python
# _base.py
class Model(pydantic.BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

# errors.py
class QuanhengError(Exception): ...
class UnconfirmedInputError(QuanhengError):
    targets: list[str]                      # 例如 ["scores.b.commute"]
class PatchError(QuanhengError): ...
class ModelFileError(QuanhengError): ...

# quantity.py
class Exact(Model):       kind: Literal["exact"] = "exact"; value: float
class Range(Model):       kind: Literal["range"] = "range"; low: float; high: float
class ThreePoint(Model):  kind: Literal["three_point"] = "three_point"; low: float; mode: float; high: float
class UserInput(Model):   type: Literal["user_input"] = "user_input"
class Quoted(Model):      type: Literal["quoted"] = "quoted"; quote: str
class Imported(Model):    type: Literal["imported"] = "imported"; file: str; location: str
class LlmInferred(Model): type: Literal["llm_inferred"] = "llm_inferred"; reason: str

class Quantity(Model):
    value: Exact | Range | ThreePoint       # 按 kind 区分
    unit: str = ""
    source: UserInput | Quoted | Imported | LlmInferred = UserInput()   # 按 type 区分
    confirmed: bool = False
    @property
    def needs_confirmation(self) -> bool    # 来源是 quoted 或 llm_inferred 且 confirmed 为 False
    @property
    def central(self) -> float              # 确定值本身；范围取中点；三点估计取 mode
    @property
    def bounds(self) -> tuple[float, float] # (最低, 最高)；确定值两者相同
    @property
    def is_uncertain(self) -> bool          # 不是 Exact
```

- 校验规则：`Range` 要求 `low < high`；`ThreePoint` 要求 `low <= mode <= high` 且 `low < high`；`Quoted.quote` 非空。
- 简写（给手写文件用，`Quantity` 的前置校验）：一个数字等于"确定值、手填"；两个数字的列表是范围；三个数字的列表是三点估计。布尔值不算数字。

- [ ] **Step 1: 写失败的测试**

```python
import pytest
from pydantic import ValidationError
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

@pytest.mark.parametrize("source, confirmed, expected", [
    ({"type": "user_input"}, False, False),
    ({"type": "imported", "file": "a.csv", "location": "B2"}, False, False),
    ({"type": "quoted", "quote": "房租7万"}, False, True),
    ({"type": "quoted", "quote": "房租7万"}, True, False),
    ({"type": "llm_inferred", "reason": "估算"}, False, True),
    ({"type": "llm_inferred", "reason": "估算"}, True, False),
])
def test_confirmation_rule(source, confirmed, expected):
    x = q(value={"kind": "exact", "value": 1}, source=source, confirmed=confirmed)
    assert x.needs_confirmation is expected

@pytest.mark.parametrize("bad", [
    [5, 5], [7, 3], [1, 0, 2], [0, 3, 2], [0, 0, 0], [1, 2, 3, 4], [],
    float("nan"), float("inf"), "70000", True, None,
])
def test_rejects_invalid_values(bad):
    with pytest.raises(ValidationError):
        Quantity.model_validate(bad)

def test_quoted_requires_quote_and_extra_fields_rejected():
    with pytest.raises(ValidationError):
        q(value={"kind": "exact", "value": 1}, source={"type": "quoted", "quote": ""})
    with pytest.raises(ValidationError):
        q(value={"kind": "exact", "value": 1}, typo=1)

def test_json_round_trip():
    x = q(value={"kind": "range", "low": 40, "high": 55}, unit="分钟",
          source={"type": "llm_inferred", "reason": "按地图估算"}, confirmed=True)
    assert Quantity.model_validate_json(x.model_dump_json()) == x
```

- [ ] **Step 2: 运行，确认失败**

Run: `uv run pytest tests/test_quantity.py -v`
Expected: 失败，`quanheng.quantity` 不存在。

- [ ] **Step 3: 按 Interfaces 实现三个文件**

- [ ] **Step 4: 运行，确认通过**

Run: `uv run pytest tests/test_quantity.py -v && uv run mypy`
Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add src/quanheng/_base.py src/quanheng/errors.py src/quanheng/quantity.py tests/test_quantity.py
git commit -m "feat: quantity with provenance and confirmation rule"
```

---

### Task 3: 独立随机流与分位换算

**Files:**
- Create: `src/quanheng/sampling.py`
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `Quantity`（任务 2）
- Produces:

```python
def quantiles(seed: int, key: str, n: int) -> NDArray[np.float64]      # [0, 1) 上的 n 个分位
def from_quantiles(q: Quantity, u: NDArray[np.float64]) -> NDArray[np.float64]
```

- [ ] **Step 1: 写失败的测试**

```python
import numpy as np
from quanheng.quantity import Quantity
from quanheng.sampling import from_quantiles, quantiles

def test_stream_is_pinned():
    # 这个测试失败意味着随机流变了：所有已保存的结果都无法复算，属于破坏性变更。
    np.testing.assert_allclose(
        quantiles(0, "score:a:rent", 3), [0.97302797, 0.41170296, 0.09270841], atol=1e-8)

def test_streams_depend_on_seed_and_key_only():
    assert np.array_equal(quantiles(7, "k", 5), quantiles(7, "k", 5))
    assert not np.array_equal(quantiles(7, "k", 5), quantiles(7, "j", 5))
    assert not np.array_equal(quantiles(7, "k", 5), quantiles(8, "k", 5))

def test_inverse_cdf_values():
    u = np.array
    assert from_quantiles(Quantity.model_validate(7), u([0.1, 0.9])).tolist() == [7, 7]
    assert from_quantiles(Quantity.model_validate([10, 20]), u([0.0, 0.25])).tolist() == [10, 12.5]
    np.testing.assert_allclose(from_quantiles(Quantity.model_validate([0, 1, 2]), u([0.5])), [1.0])
    np.testing.assert_allclose(from_quantiles(Quantity.model_validate([0, 0, 4]), u([0.75])), [2.0])
    np.testing.assert_allclose(from_quantiles(Quantity.model_validate([0, 3, 4]), u([0.5])), [6 ** 0.5])

def test_inverse_cdf_is_monotone_and_within_bounds():
    u = np.sort(quantiles(1, "m", 1000))
    for spec in ([10, 20], [0, 1, 2], [0, 0, 4], [-5, -1, 0]):
        x = from_quantiles(Quantity.model_validate(spec), u)
        assert np.all(np.diff(x) >= 0) and x.min() >= spec[0] and x.max() <= spec[-1]
```

- [ ] **Step 2: 运行，确认失败**

Run: `uv run pytest tests/test_sampling.py -v`
Expected: 失败，`quanheng.sampling` 不存在。

- [ ] **Step 3: 实现**

`quantiles` 的算法必须逐步照做，否则 `test_stream_is_pinned` 不会通过：

```python
def quantiles(seed, key, n):
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    words = [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4)]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, *words])))
    return rng.random(n)
```

`from_quantiles`：确定值返回常数数组；范围是 `low + u * (high - low)`；三点估计是三角分布的反函数，设 `c = (mode - low) / (high - low)`，`u < c` 时为 `low + sqrt(u * (high - low) * (mode - low))`，否则为 `high - sqrt((1 - u) * (high - low) * (high - mode))`。

- [ ] **Step 4: 运行，确认通过**

Run: `uv run pytest tests/test_sampling.py -v && uv run mypy`
Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add src/quanheng/sampling.py tests/test_sampling.py
git commit -m "feat: per-input random streams and inverse CDF sampling"
```

---

### Task 4: 权重

**Files:**
- Create: `src/quanheng/weights.py`
- Test: `tests/test_weights.py`

**Interfaces:**
- Consumes: `Model`（任务 2）、`quantiles`（任务 3）
- Produces:

```python
Certainty = Literal["fixed", "high", "medium", "low"]
SIGMA: dict[Certainty, float] = {"fixed": 0.0, "high": 0.1, "medium": 0.25, "low": 0.5}
POOR_CONSISTENCY = 0.1          # 一致性指标高于它算"一致性差"
POOR_CONSISTENCY_WIDENING = 1.5

class PairwiseJudgment(Model):
    a: str; b: str
    ratio: float                # a 的变化幅度是 b 的 ratio 倍重要，1/9 <= ratio <= 9
class PairwiseResult(Model):
    weights: dict[str, float]   # 合计为 1
    consistency_ratio: float
    most_inconsistent: tuple[str, str] | None
class WeightSpec(Model):
    values: dict[str, float]    # 每个 >= 0，合计 > 0，任意量纲（0–100 打分可直接填）
    certainty: Certainty = "medium"
    consistency_ratio: float | None = None
    pairwise: list[PairwiseJudgment] | None = None   # 只作记录，计算不使用
    def normalized(self) -> dict[str, float]         # 合计为 1，保持键的顺序
    def sigma(self) -> float    # SIGMA[certainty]；consistency_ratio > 0.1 时再乘 1.5

def weights_from_pairwise(criteria: list[str], judgments: list[PairwiseJudgment]) -> PairwiseResult
def rescale_for_anchor_change(weights: dict[str, float], criterion: str,
                              old: tuple[float, float], new: tuple[float, float]) -> dict[str, float]
def noise_factors(seed: int, criteria: list[str], sigma: float, n: int) -> NDArray[np.float64]
```

- 引擎多一档 `"fixed"`（权重完全不波动），界面只提供设计文档里的三档。它用于手算核对和测试。
- `old`、`new` 都是 `(差, 好)`。

- [ ] **Step 1: 写失败的测试**

```python
import numpy as np, pytest
from pydantic import ValidationError
from quanheng.weights import (PairwiseJudgment as J, WeightSpec, noise_factors,
                              rescale_for_anchor_change, weights_from_pairwise)

def test_consistent_pairwise():
    r = weights_from_pairwise(["a", "b", "c"],
        [J(a="a", b="b", ratio=2), J(a="b", b="c", ratio=3), J(a="a", b="c", ratio=6)])
    assert r.weights == pytest.approx({"a": 0.6, "b": 0.3, "c": 0.1})
    assert r.consistency_ratio == pytest.approx(0, abs=1e-9) and r.most_inconsistent is None

def test_inconsistent_pair_is_named():
    js = [J(a="a", b="b", ratio=2), J(a="a", b="c", ratio=4), J(a="a", b="d", ratio=8),
          J(a="b", b="c", ratio=2), J(a="b", b="d", ratio=4), J(a="c", b="d", ratio=0.25)]
    r = weights_from_pairwise(list("abcd"), js)
    assert r.consistency_ratio == pytest.approx(0.2118, abs=1e-3)
    assert r.most_inconsistent == ("c", "d")
    assert r.weights == pytest.approx({"a": 0.5380, "b": 0.2690, "c": 0.0800, "d": 0.1131}, abs=1e-3)

def test_reversed_judgment_and_two_criteria():
    r1 = weights_from_pairwise(["a", "b"], [J(a="a", b="b", ratio=2)])
    r2 = weights_from_pairwise(["a", "b"], [J(a="b", b="a", ratio=0.5)])
    assert r1.weights == pytest.approx({"a": 2 / 3, "b": 1 / 3}) == r2.weights
    assert r1.consistency_ratio == 0

def test_incomplete_or_duplicate_judgments_rejected():
    with pytest.raises(ValueError):
        weights_from_pairwise(["a", "b", "c"], [J(a="a", b="b", ratio=2)])
    with pytest.raises(ValueError):
        weights_from_pairwise(["a", "b"], [J(a="a", b="b", ratio=2), J(a="b", b="a", ratio=1)])

@pytest.mark.parametrize("certainty, cr, expected", [
    ("fixed", None, 0.0), ("high", None, 0.1), ("medium", None, 0.25), ("low", None, 0.5),
    ("medium", 0.1, 0.25), ("medium", 0.2, 0.375),
])
def test_sigma(certainty, cr, expected):
    w = WeightSpec(values={"a": 1}, certainty=certainty, consistency_ratio=cr)
    assert w.sigma() == pytest.approx(expected)

def test_normalized_and_validation():
    assert WeightSpec(values={"a": 30, "b": 10}).normalized() == {"a": 0.75, "b": 0.25}
    for bad in ({"a": -1, "b": 2}, {"a": 0, "b": 0}, {}):
        with pytest.raises(ValidationError):
            WeightSpec(values=bad)

def test_rescale_keeps_exchange_rate():
    w = rescale_for_anchor_change(
        {"rent": 0.30, "commute": 0.35, "light": 0.15, "size": 0.20},
        "rent", old=(90000, 60000), new=(100000, 55000))
    assert w == pytest.approx(
        {"rent": 0.391304, "commute": 0.304348, "light": 0.130435, "size": 0.173913}, abs=1e-6)
    with pytest.raises(ValueError):   # 方向反了
        rescale_for_anchor_change({"rent": 1.0}, "rent", old=(90000, 60000), new=(55000, 100000))

def test_noise_factors():
    assert np.array_equal(noise_factors(0, ["a", "b"], 0.0, 100), np.ones((100, 2)))
    f = noise_factors(0, ["a", "b"], 0.25, 1000)
    assert f.shape == (1000, 2) and np.all(f > 0) and np.all(np.isfinite(f))
    assert np.array_equal(f[:, 1], noise_factors(0, ["b"], 0.25, 1000)[:, 0])   # 各准则的流互不影响
```

- [ ] **Step 2: 运行，确认失败**

Run: `uv run pytest tests/test_weights.py -v`
Expected: 失败，`quanheng.weights` 不存在。

- [ ] **Step 3: 实现**

`weights_from_pairwise` 的方法：
1. 建立 n×n 矩阵 A，`A[i][j] = ratio`，`A[j][i] = 1 / ratio`，对角线为 1。每一对准则必须恰好有一条判断，否则 `ValueError`。
2. 权重是各行的几何平均数，归一化。
3. `λ = mean((A @ w) / w)`，一致性指标 `CR = (λ - n) / (n - 1) / RI[n]`；n ≤ 2 时 CR 为 0。`RI = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}`；n > 10 时 `ValueError`。
4. `CR <= 0.1` 时 `most_inconsistent` 为 `None`；否则是 `|ln(A[i][j] * w[j] / w[i])|` 最大的那一对，并列时取按准则顺序排在最前的。

`rescale_for_anchor_change`：该准则的权重乘以 `(new_好 - new_差) / (old_好 - old_差)`，然后全部归一化。比值 ≤ 0 时 `ValueError`。

`noise_factors`：第 j 列是 `exp(sigma * ndtri(clip(quantiles(seed, f"weight:{id}", n), 2**-53, 1 - 2**-53)))`，其中 `ndtri` 来自 `scipy.special`。

- [ ] **Step 4: 运行，确认通过**

Run: `uv run pytest tests/test_weights.py -v && uv run mypy`
Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add src/quanheng/weights.py tests/test_weights.py
git commit -m "feat: weights from pairwise comparison, anchor rescaling, proportional noise"
```

---

### Task 5: 选择模型的结构和示例文件

**Files:**
- Create: `src/quanheng/choice/__init__.py`, `src/quanheng/choice/model.py`, `examples/rent-a-or-b.json`, `tests/conftest.py`
- Test: `tests/test_choice_model.py`

**Interfaces:**
- Consumes: `Model`、`Quantity`（任务 2）、`WeightSpec`（任务 4）
- Produces:

```python
Id = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]+$")]

class Option(Model):     id: Id; name: str; note: str = ""
class Criterion(Model):  id: Id; name: str; unit: str = ""; bad: float; good: float; cap_at_good: bool = False
class Constraint(Model): criterion: str; op: Literal["<=", ">="]; threshold: float
class ChoiceModel(Model):
    kind: Literal["choice"] = "choice"
    schema_version: Literal[1] = 1
    title: str
    description: str = ""
    options: list[Option]
    criteria: list[Criterion]
    constraints: list[Constraint] = []
    scores: dict[str, dict[str, Quantity]]     # 候选 id → 准则 id → 数值
    weights: WeightSpec
    def unconfirmed_inputs(self) -> list[str]  # 形如 "scores.b.commute"，按候选、准则的顺序
```

- 夹具（`tests/conftest.py`）：`hand_model() -> ChoiceModel`、`rent_model() -> ChoiceModel`（读示例文件）、`hand_dict() -> dict`（供改动后再校验）。
- 校验规则，任何一条不满足都是 `ValidationError`：候选至少 2 个；准则至少 1 条；候选 id、准则 id 各自不重复；`bad != good`；`scores` 的候选和准则与声明的完全一致，不多不少；约束引用的准则存在；`weights.values` 的键与准则 id 完全一致。
- 跨字段的校验（`scores` 不完整、约束或权重引用了不存在的准则）在错误信息里写明位置，形如 `scores.b 缺少准则 size`。任务 12 的命令行靠它告诉使用者错在哪。

- [ ] **Step 1: 写示例文件和夹具**

`examples/rent-a-or-b.json`：

```json
{
  "kind": "choice",
  "schema_version": 1,
  "title": "租 A 还是租 B？",
  "description": "两套房子二选一的示例。",
  "options": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
  "criteria": [
    {"id": "rent", "name": "房租", "unit": "日元/月", "bad": 90000, "good": 60000},
    {"id": "commute", "name": "通勤", "unit": "分钟", "bad": 60, "good": 20},
    {"id": "light", "name": "采光", "unit": "分", "bad": 0, "good": 10},
    {"id": "size", "name": "面积", "unit": "平方米", "bad": 20, "good": 35}
  ],
  "constraints": [{"criterion": "rent", "op": "<=", "threshold": 85000}],
  "scores": {
    "a": {"rent": 78000, "commute": [25, 35], "light": [6, 8], "size": 25},
    "b": {
      "rent": 70000,
      "commute": {
        "value": {"kind": "range", "low": 40, "high": 55},
        "unit": "分钟",
        "source": {"type": "llm_inferred", "reason": "按地图估算，未实测"},
        "confirmed": true
      },
      "light": [4, 5, 7],
      "size": 28
    }
  },
  "weights": {"values": {"rent": 30, "commute": 35, "light": 15, "size": 20}, "certainty": "medium"}
}
```

`tests/conftest.py` 里的手算模型（后续任务的期望值都来自它）：

```python
HAND = {
    "title": "hand",
    "options": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
    "criteria": [
        {"id": "rent", "name": "房租", "unit": "日元/月", "bad": 90000, "good": 50000},
        {"id": "commute", "name": "通勤", "unit": "分钟", "bad": 60, "good": 20},
    ],
    "scores": {"a": {"rent": 70000, "commute": 30}, "b": {"rent": 60000, "commute": 50}},
    "weights": {"values": {"rent": 1, "commute": 1}, "certainty": "fixed"},
}
```

手算：A 的总分是 0.5×0.5 + 0.5×0.75 = 0.625，B 是 0.5×0.75 + 0.5×0.25 = 0.5。

- [ ] **Step 2: 写失败的测试**

```python
import copy, pytest
from pydantic import ValidationError
from quanheng.choice.model import ChoiceModel

def test_example_and_hand_model_load(rent_model, hand_model):
    assert rent_model.title == "租 A 还是租 B？" and len(rent_model.criteria) == 4
    assert hand_model.weights.normalized() == {"rent": 0.5, "commute": 0.5}
    assert rent_model.unconfirmed_inputs() == [] == hand_model.unconfirmed_inputs()

def test_unconfirmed_inputs_are_listed(rent_model):
    data = rent_model.model_dump()
    data["scores"]["b"]["commute"]["confirmed"] = False
    assert ChoiceModel.model_validate(data).unconfirmed_inputs() == ["scores.b.commute"]

def mutate(d, fn):
    d = copy.deepcopy(d); fn(d); return d

@pytest.mark.parametrize("fn", [
    lambda d: d["options"].pop(),                                   # 只剩一个候选
    lambda d: d.update(criteria=[]),                                # 没有准则
    lambda d: d["options"].append({"id": "a", "name": "重复"}),       # 候选 id 重复
    lambda d: d["options"][0].update(id="a.b"),                     # id 含非法字符
    lambda d: d["criteria"][0].update(good=90000),                  # 差和好相等
    lambda d: d["scores"]["b"].pop("commute"),                      # 缺一个数值
    lambda d: d["scores"].update(c={"rent": 1, "commute": 1}),      # 多一个候选的数值
    lambda d: d.update(constraints=[{"criterion": "nope", "op": "<=", "threshold": 1}]),
    lambda d: d["weights"]["values"].pop("commute"),                # 权重缺一条
    lambda d: d.update(schema_version=2),
])
def test_invalid_models_rejected(hand_dict, fn):
    with pytest.raises(ValidationError):
        ChoiceModel.model_validate(mutate(hand_dict, fn))
```

- [ ] **Step 3: 运行，确认失败**

Run: `uv run pytest tests/test_choice_model.py -v`
Expected: 失败，`quanheng.choice.model` 不存在。

- [ ] **Step 4: 按 Interfaces 实现 `model.py`**

- [ ] **Step 5: 运行，确认通过**

Run: `uv run pytest tests/test_choice_model.py -v && uv run mypy`
Expected: 全部通过。

- [ ] **Step 6: 提交**

```bash
git add src/quanheng/choice examples/rent-a-or-b.json tests/conftest.py tests/test_choice_model.py
git commit -m "feat: choice model schema with validation and example decision"
```

---

### Task 6: 抽样与逐次评估

**Files:**
- Create: `src/quanheng/choice/evaluate.py`
- Test: `tests/test_choice_evaluate.py`

**Interfaces:**
- Consumes: `ChoiceModel`（任务 5）、`quantiles`、`from_quantiles`（任务 3）、`noise_factors`、`rescale_for_anchor_change`（任务 4）、`UnconfirmedInputError`（任务 2）
- Produces:

```python
TIE_TOLERANCE = 1e-12
MIN_SAMPLES, MAX_SAMPLES = 100, 1_000_000

@dataclass(frozen=True)
class Draws:
    raw: NDArray[np.float64]              # (抽样数, 候选数, 准则数)，抽到的原始数值
    quantile: NDArray[np.float64]         # 同形状，对应的分位
    weight_factor: NDArray[np.float64]    # (抽样数, 准则数)，权重的随机系数
    weight_quantile: NDArray[np.float64]  # (抽样数, 准则数)，权重系数对应的分位

@dataclass(frozen=True)
class Outcome:
    scores: NDArray[np.float64]           # (抽样数, 候选数)
    feasible: NDArray[np.bool_]           # (抽样数, 候选数)
    share: NDArray[np.float64]            # (抽样数, 候选数)；每行合计为 1，无人可行时合计为 0
    none_feasible: NDArray[np.bool_]      # (抽样数,)

def central_weights(model: ChoiceModel) -> NDArray[np.float64]   # 归一化，按准则顺序
def draw(model: ChoiceModel, samples: int, seed: int) -> Draws
def evaluate(model: ChoiceModel, raw: NDArray[np.float64],
             weights: NDArray[np.float64], weight_factor: NDArray[np.float64]) -> Outcome
```

固定的规则：
- 随机流的标识：数值是 `score:{候选 id}:{准则 id}`，权重是 `weight:{准则 id}`。
- `draw` 是引擎的入口：`model.unconfirmed_inputs()` 非空时抛 `UnconfirmedInputError`；`samples` 不在 100 到 1,000,000 之间时抛 `ValueError`。
- 分数换算：`v = (x - bad) / (good - bad)`，不截断；`cap_at_good` 为真时 `v = min(v, 1)`。
- 每次抽样的权重：`weights * weight_factor`，按行归一化。总分是权重与分数的加权和。
- 可行：该候选在这次抽样里满足全部约束（约束作用在原始数值上）。
- 胜出份额：在可行的候选里取最高分；与最高分相差不超过 `TIE_TOLERANCE` 的候选平分这一次。

- [ ] **Step 1: 写失败的测试**

```python
import numpy as np, pytest
from quanheng.choice.evaluate import central_weights, draw, evaluate
from quanheng.choice.model import ChoiceModel
from quanheng.errors import UnconfirmedInputError
from quanheng.weights import rescale_for_anchor_change

def run(model, samples=10_000, seed=0):
    d = draw(model, samples, seed)
    return d, evaluate(model, d.raw, central_weights(model), d.weight_factor)

def patched(model, fn):
    data = model.model_dump(); fn(data); return ChoiceModel.model_validate(data)

def test_hand_model_scores(hand_model):
    _, out = run(hand_model, samples=100)
    assert np.allclose(out.scores, [0.625, 0.5]) and np.all(out.share == [1.0, 0.0])

def test_identical_options_split_ties(hand_model):
    two = patched(hand_model, lambda d: d["scores"].update(b=d["scores"]["a"]))
    assert np.all(run(two, samples=100)[1].share == 0.5)
    three = patched(two, lambda d: (d["options"].append({"id": "c", "name": "C"}),
                                     d["scores"].update(c=d["scores"]["a"])))
    assert np.allclose(run(three, samples=100)[1].share, 1 / 3)

def test_shares_plus_none_feasible_sum_to_one(rent_model):
    _, out = run(rent_model)
    assert np.allclose(out.share.sum(axis=1) + out.none_feasible, 1.0)

def test_constraints_are_judged_per_sample(hand_model):
    m = patched(hand_model, lambda d: (d["scores"]["a"].update(rent=[70000, 90000]),
        d.update(constraints=[{"criterion": "rent", "op": "<=", "threshold": 85000}])))
    _, out = run(m)
    assert out.feasible[:, 0].mean() == pytest.approx(0.75, abs=0.015)
    assert out.feasible[:, 1].all()
    assert np.all(out.share[~out.feasible[:, 0], 0] == 0)

def test_nobody_feasible(hand_model):
    m = patched(hand_model, lambda d: d.update(
        constraints=[{"criterion": "rent", "op": "<=", "threshold": 50000}]))
    _, out = run(m, samples=100)
    assert out.none_feasible.all() and np.all(out.share == 0)

def test_strict_dominance_always_wins(hand_model):
    m = patched(hand_model, lambda d: d["scores"].update(
        a={"rent": [55000, 60000], "commute": [20, 25]},
        b={"rent": [65000, 80000], "commute": [30, 55]}))
    m = patched(m, lambda d: d["weights"].update(certainty="low"))
    assert np.all(run(m)[1].share[:, 0] == 1.0)

def test_anchor_change_with_rescaled_weights_keeps_every_winner(rent_model):
    _, before = run(rent_model)
    def change(d):
        d["criteria"][0].update(bad=100000, good=55000)
        d["weights"]["values"] = rescale_for_anchor_change(
            rent_model.weights.normalized(), "rent", old=(90000, 60000), new=(100000, 55000))
    _, after = run(patched(rent_model, change))
    assert np.array_equal(before.share, after.share)

def test_cap_at_good(hand_model):
    far = patched(hand_model, lambda d: (d["scores"]["a"].update(commute=10),
                                         d["criteria"][1].update(cap_at_good=True)))
    at_good = patched(hand_model, lambda d: d["scores"]["a"].update(commute=20))
    assert run(far, 100)[1].scores[0, 0] == run(at_good, 100)[1].scores[0, 0]
    uncapped = patched(hand_model, lambda d: d["scores"]["a"].update(commute=10))
    assert run(uncapped, 100)[1].scores[0, 0] > run(at_good, 100)[1].scores[0, 0]

def test_changing_one_input_leaves_other_draws_untouched(rent_model):
    before = draw(rent_model, 1000, 0)
    after = draw(patched(rent_model, lambda d: d["scores"]["a"].update(commute=[30, 50])), 1000, 0)
    mask = np.ones(before.raw.shape[1:], bool); mask[0, 1] = False
    assert np.array_equal(before.raw[:, mask], after.raw[:, mask])
    assert np.array_equal(before.quantile, after.quantile)
    assert np.array_equal(before.weight_factor, after.weight_factor)

def test_unconfirmed_input_is_refused(rent_model):
    m = patched(rent_model, lambda d: d["scores"]["b"]["commute"].update(confirmed=False))
    with pytest.raises(UnconfirmedInputError) as e:
        draw(m, 1000, 0)
    assert e.value.targets == ["scores.b.commute"]

@pytest.mark.parametrize("samples", [0, -5, 99, 1_000_001])
def test_sample_count_bounds(hand_model, samples):
    with pytest.raises(ValueError):
        draw(hand_model, samples, 0)
```

- [ ] **Step 2: 运行，确认失败**

Run: `uv run pytest tests/test_choice_evaluate.py -v`
Expected: 失败，`quanheng.choice.evaluate` 不存在。

- [ ] **Step 3: 按 Interfaces 和固定规则实现**

`evaluate` 要全程用数组运算，不对抽样写 Python 循环。

- [ ] **Step 4: 运行，确认通过**

Run: `uv run pytest tests/test_choice_evaluate.py -v && uv run mypy`
Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add src/quanheng/choice/evaluate.py tests/test_choice_evaluate.py
git commit -m "feat: sampling and per-sample evaluation for the choice model"
```

---

### Task 7: 汇总成结果

**Files:**
- Create: `src/quanheng/choice/result.py`, `src/quanheng/choice/solve.py`
- Test: `tests/test_choice_solve.py`

**Interfaces:**
- Consumes: `draw`、`evaluate`、`central_weights`（任务 6）
- Produces:

```python
# result.py
class Thresholds(Model):  clear: float = 0.90; leaning: float = 0.70
class Meta(Model):        seed: int; samples: int; engine_version: str; numpy_version: str
class Verdict(Model):
    level: Literal["clear", "leaning", "toss_up", "mostly_infeasible"]
    winner: str | None; runner_up: str | None; win_rate: float; std_error: float
class Margin(Model):      p5: float; p50: float; p95: float; used_share: float
                          bin_edges: list[float]; counts: list[int]      # 20 个区间
class ConstraintGap(Model): index: int; criterion: str; nearest_option: str; gap: float
class ResultWarning(Model):
    code: Literal["outside_anchors", "central_disagrees", "poor_consistency"]
    target: str | None = None
class TippingPoint(Model):
    target: str; current: float; flips_at: float | None
    new_leader: str | None; within_stated_range: bool
class InfoValue(Model):
    target: str; share: float; leader_win_rate_low: float; leader_win_rate_high: float
class ChoiceResult(Model):
    kind: Literal["choice_result"] = "choice_result"
    meta: Meta
    verdict: Verdict
    win_rates: dict[str, float]
    none_feasible_share: float
    feasibility: dict[str, float]
    central_scores: dict[str, float]
    contributions: dict[str, dict[str, float]]
    central_leader: str | None
    margin: Margin | None
    tipping_points: list[TippingPoint] = []
    info_values: list[InfoValue] = []
    constraint_gaps: list[ConstraintGap]
    warnings: list[ResultWarning]
    assumptions: list[str]

# solve.py
def solve_choice(model: ChoiceModel, *, samples: int = 10_000, seed: int = 0,
                 thresholds: Thresholds = Thresholds()) -> ChoiceResult
```

固定的规则：
- `win_rates` 是 `share` 按列的平均。领先者是胜出率最高的候选，第二名是次高的，并列时取排在前面的候选。
- 档位：`none_feasible_share` 严格大于最高胜出率时为 `mostly_infeasible`，此时 `winner`、`runner_up` 为 `None`，`win_rate` 和 `std_error` 为 0。否则按 `thresholds` 判定，`winner` 是领先者。
- `std_error = sqrt(p * (1 - p) / samples)`。
- 中心值计算：每个数值取 `central`，权重取 `central_weights`。`contributions[候选][准则]` 是权重乘分数，`central_scores` 是其合计。`central_leader` 是在中心值下可行的候选里总分最高的，没有可行的则为 `None`。
- `margin`：只统计领先者和第二名都可行的抽样，分差为领先者减第二名；分位用 `np.quantile` 的默认方法；直方图用 `np.histogram(d, bins=20)`。`winner` 为 `None` 或没有可用的抽样时为 `None`。
- `constraint_gaps`：对每条约束，用中心值算每个候选违反了多少（满足则为 0），取最小的那个候选和它的差距。
- `warnings`：数值的 `bounds` 超出两个锚点之间的区间时加 `outside_anchors`，`target` 是该数值的路径；`winner` 不为 `None` 且与 `central_leader` 不同时加 `central_disagrees`；`weights.consistency_ratio > 0.1` 时加 `poor_consistency`。
- `assumptions` 固定为 `["independent_inputs", "linear_value", "uniform_range", "triangular_three_point", "proportional_weight_noise"]`。
- 本任务里 `tipping_points` 和 `info_values` 保持为空，任务 8、9 填入。

- [ ] **Step 1: 写失败的测试**

```python
import numpy, pytest, quanheng
from quanheng.choice.model import ChoiceModel
from quanheng.choice.result import Thresholds
from quanheng.choice.solve import solve_choice

def patched(model, fn):
    data = model.model_dump(); fn(data); return ChoiceModel.model_validate(data)

def one_criterion(a, b, certainty="fixed"):
    return ChoiceModel.model_validate({
        "title": "t", "options": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
        "criteria": [{"id": "q", "name": "Q", "bad": 0, "good": 10}],
        "scores": {"a": {"q": a}, "b": {"q": b}},
        "weights": {"values": {"q": 1}, "certainty": certainty}})

def test_hand_model_result(hand_model):
    r = solve_choice(hand_model)
    assert (r.verdict.level, r.verdict.winner, r.verdict.runner_up) == ("clear", "a", "b")
    assert r.verdict.win_rate == 1.0 and r.verdict.std_error == 0.0
    assert r.win_rates == {"a": 1.0, "b": 0.0} and r.none_feasible_share == 0.0
    assert r.feasibility == {"a": 1.0, "b": 1.0}
    assert r.central_scores == pytest.approx({"a": 0.625, "b": 0.5})
    assert r.contributions["a"] == pytest.approx({"rent": 0.25, "commute": 0.375})
    assert r.contributions["b"] == pytest.approx({"rent": 0.375, "commute": 0.125})
    assert r.central_leader == "a" and r.warnings == [] and r.constraint_gaps == []
    assert r.margin.p50 == pytest.approx(0.125) and r.margin.used_share == 1.0
    assert len(r.margin.counts) == 20 and sum(r.margin.counts) == 10_000
    assert r.meta.model_dump() == {"seed": 0, "samples": 10_000,
        "engine_version": quanheng.__version__, "numpy_version": numpy.__version__}
    assert r.assumptions == ["independent_inputs", "linear_value", "uniform_range",
                             "triangular_three_point", "proportional_weight_noise"]

def test_known_win_rate_and_thresholds():
    m = one_criterion(5, [0, 50 / 7])            # B 超过 5 的概率正好是 30%
    r = solve_choice(m)
    assert r.win_rates["a"] == pytest.approx(0.70, abs=0.015) and r.verdict.level == "leaning"
    assert r.verdict.std_error == pytest.approx(0.0046, abs=2e-4)
    assert solve_choice(m, thresholds=Thresholds(clear=0.6)).verdict.level == "clear"
    assert solve_choice(m, thresholds=Thresholds(leaning=0.75)).verdict.level == "toss_up"

def test_win_rates_and_none_feasible_sum_to_one(rent_model):
    r = solve_choice(rent_model)
    assert sum(r.win_rates.values()) + r.none_feasible_share == pytest.approx(1.0)
    assert r.verdict.level == "leaning" and r.verdict.winner == "a"
    assert 0.78 < r.verdict.win_rate < 0.84

def test_mostly_infeasible(hand_model):
    m = patched(hand_model, lambda d: d.update(
        constraints=[{"criterion": "rent", "op": "<=", "threshold": 50000}]))
    r = solve_choice(m)
    assert r.verdict.level == "mostly_infeasible" and r.verdict.winner is None
    assert r.none_feasible_share == 1.0 and r.margin is None and r.central_leader is None
    assert [g.model_dump() for g in r.constraint_gaps] == [
        {"index": 0, "criterion": "rent", "nearest_option": "b", "gap": 10000.0}]
    assert r.tipping_points == [] and r.info_values == []

def test_same_inputs_give_identical_bytes(rent_model):
    assert solve_choice(rent_model).model_dump_json() == solve_choice(rent_model).model_dump_json()
    assert solve_choice(rent_model, seed=1).win_rates != solve_choice(rent_model).win_rates

def test_outside_anchor_warning(hand_model):
    r = solve_choice(patched(hand_model, lambda d: d["scores"]["a"].update(rent=95000)))
    assert [w.model_dump() for w in r.warnings] == [
        {"code": "outside_anchors", "target": "scores.a.rent"}]

def test_central_values_can_disagree_with_sampling():
    r = solve_choice(one_criterion([0, 0, 10], 2))    # A 最可能是 0，但有 64% 的情况超过 2
    assert r.central_leader == "b" and r.verdict.winner == "a"
    assert r.win_rates["a"] == pytest.approx(0.64, abs=0.015)
    assert "central_disagrees" in [w.code for w in r.warnings]

def test_poor_consistency_warning(hand_model):
    r = solve_choice(patched(hand_model, lambda d: d["weights"].update(consistency_ratio=0.2)))
    assert "poor_consistency" in [w.code for w in r.warnings]
```

- [ ] **Step 2: 运行，确认失败**

Run: `uv run pytest tests/test_choice_solve.py -v`
Expected: 失败，`quanheng.choice.solve` 不存在。

- [ ] **Step 3: 按 Interfaces 和固定规则实现 `result.py` 与 `solve.py`**

- [ ] **Step 4: 运行，确认通过**

Run: `uv run pytest tests/test_choice_solve.py -v && uv run mypy`
Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add src/quanheng/choice/result.py src/quanheng/choice/solve.py tests/test_choice_solve.py
git commit -m "feat: assemble choice result with verdict, margin, warnings and metadata"
```

---

### Task 8: 翻盘点

**Files:**
- Create: `src/quanheng/choice/tipping.py`
- Modify: `src/quanheng/choice/solve.py`（`winner` 不为 `None` 时填入 `tipping_points`）
- Test: `tests/test_choice_tipping.py`

**Interfaces:**
- Consumes: `Draws`、`evaluate`、`central_weights`（任务 6）、`TippingPoint`（任务 7）
- Produces:

```python
GRID_POINTS = 101
BISECTION_STEPS = 40

def leader_with_pinned(model: ChoiceModel, draws: Draws, target: str, value: float) -> str
    # 把 target 固定在 value、其余照常抽样时，胜出率最高的候选的 id
def tipping_points(model: ChoiceModel, draws: Draws, winner: str) -> list[TippingPoint]
```

目标的写法：数值是 `scores.{候选 id}.{准则 id}`，权重是 `weights.{准则 id}`。

固定的规则：
- 固定一个数值：把 `draws.raw` 里该候选该准则的那一列全部换成 `value`，再 `evaluate`。
- 固定一条权重：中心权重里该条设为 `value`，其余各条乘以 `(1 - value) / (1 - 原值)`；权重的随机系数不变。
- 搜索范围：数值在两个锚点之间；权重在 0 到 1 之间。
- 权重目标只在准则多于一条、且该权重的中心值小于 1 时才计算。
- 搜索步骤，`current` 是该输入的中心值：
  1. 固定在 `current` 时领先者已经不是 `winner`：`flips_at = current`。
  2. 否则在搜索范围内取 101 个等距点，找出领先者不是 `winner` 的点。一个都没有：`flips_at = None`。
  3. 否则取其中离 `current` 最近的点，在 `current` 和它之间做 40 次二分，`flips_at` 取最后区间的中点，`new_leader` 是那个点上的领先者。
- `within_stated_range`：数值看 `flips_at` 是否落在该数值的 `bounds` 之内；权重看是否落在 `[w * exp(-2σ), w * exp(2σ)]` 之内，`σ` 是 `model.weights.sigma()`。`flips_at` 为 `None` 时为 `False`。
- 排序：`within_stated_range` 为真的在前；其次按 `|flips_at - current|` 除以搜索范围的宽度从小到大；`flips_at` 为 `None` 的排最后；其余保持原有顺序。原有顺序是：先数值（按候选、再按准则的声明顺序），后权重（按准则的声明顺序）。

- [ ] **Step 1: 写失败的测试**

```python
import pytest
from quanheng.choice.evaluate import draw
from quanheng.choice.model import ChoiceModel
from quanheng.choice.solve import solve_choice
from quanheng.choice.tipping import leader_with_pinned

def by_target(result):
    return {t.target: t for t in result.tipping_points}

def test_hand_model_tipping_points(hand_model):
    tp = by_target(solve_choice(hand_model))
    assert tp["scores.b.commute"].flips_at == pytest.approx(40, abs=1e-3)
    assert tp["scores.b.commute"].new_leader == "b" and tp["scores.b.commute"].current == 50
    assert tp["scores.a.rent"].flips_at == pytest.approx(80000, abs=0.1)
    assert tp["scores.a.commute"].flips_at == pytest.approx(40, abs=1e-3)
    assert tp["scores.b.rent"].flips_at is None and tp["scores.b.rent"].new_leader is None
    assert tp["weights.rent"].flips_at == pytest.approx(2 / 3, abs=1e-6)
    assert tp["weights.commute"].flips_at == pytest.approx(1 / 3, abs=1e-6)

def test_crossing_a_tipping_point_changes_the_leader(rent_model):
    result = solve_choice(rent_model)
    draws = draw(rent_model, 10_000, 0)
    checked = 0
    for t in result.tipping_points:
        if t.flips_at is None or t.flips_at == t.current:
            continue
        step = (t.flips_at - t.current) * 1e-3
        assert leader_with_pinned(rent_model, draws, t.target, t.flips_at - step) == "a"
        assert leader_with_pinned(rent_model, draws, t.target, t.flips_at + step) == t.new_leader != "a"
        checked += 1
    assert checked >= 8

def test_stated_range_and_ordering(rent_model):
    tps = solve_choice(rent_model).tipping_points
    tp = {t.target: t for t in tps}
    assert tp["scores.b.commute"].flips_at == pytest.approx(41.11, abs=0.01)
    assert tp["scores.b.commute"].within_stated_range is True       # 填的是 40–55
    assert tp["scores.a.commute"].within_stated_range is False      # 约 36.2，填的是 25–35
    assert tp["weights.light"].flips_at is None
    assert {t.target for t in tps[:3]} == {"weights.commute", "weights.rent", "scores.b.commute"}
    assert tps[-1].flips_at is None

def test_single_criterion_and_extreme_weights():
    base = {"title": "t", "options": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]}
    one = ChoiceModel.model_validate({**base,
        "criteria": [{"id": "q", "name": "Q", "bad": 0, "good": 10}],
        "scores": {"a": {"q": 6}, "b": {"q": [3, 5]}},
        "weights": {"values": {"q": 1}}})
    targets = {t.target for t in solve_choice(one).tipping_points}
    assert targets == {"scores.a.q", "scores.b.q"}                  # 没有权重目标
    two = ChoiceModel.model_validate({**base,
        "criteria": [{"id": "x", "name": "X", "bad": 0, "good": 10},
                     {"id": "y", "name": "Y", "bad": 0, "good": 10}],
        "scores": {"a": {"x": 6, "y": 1}, "b": {"x": 4, "y": 9}},
        "weights": {"values": {"x": 1, "y": 0}}})
    tp = {t.target: t for t in solve_choice(two).tipping_points}
    assert "weights.x" not in tp                                    # 权重已经是 1，跳过
    assert tp["weights.y"].flips_at == pytest.approx(0.2, abs=0.02)
```

最后一条断言的手算：固定 `w_y = v` 时 `w_x = 1 - v`，A 减 B 的分差是 `0.2(1 - v) - 0.8v = 0.2 - v`，在 `v = 0.2` 处变号。这个模型的权重把握是默认的"一般"，权重系数是随机的，所以容差是 `0.02`。

- [ ] **Step 2: 运行，确认失败**

Run: `uv run pytest tests/test_choice_tipping.py -v`
Expected: 失败，`quanheng.choice.tipping` 不存在。

- [ ] **Step 3: 按固定规则实现，并在 `solve_choice` 里调用**

`solve_choice` 要把同一份 `Draws` 传进来，不为翻盘点重新抽样。

- [ ] **Step 4: 运行，确认通过**

Run: `uv run pytest tests/test_choice_tipping.py tests/test_choice_solve.py -v && uv run mypy`
Expected: 全部通过。示例决策的 `solve_choice` 在几秒内完成。

- [ ] **Step 5: 提交**

```bash
git add src/quanheng/choice/tipping.py src/quanheng/choice/solve.py tests/test_choice_tipping.py
git commit -m "feat: tipping points using the same criterion as the verdict"
```

---

### Task 9: 最值得补的信息

**Files:**
- Create: `src/quanheng/choice/info_value.py`
- Modify: `src/quanheng/choice/solve.py`（`winner` 不为 `None` 时填入 `info_values`）
- Test: `tests/test_choice_info_value.py`

**Interfaces:**
- Consumes: `Draws`、`Outcome`（任务 6）、`InfoValue`（任务 7）
- Produces:

```python
BINS = 10
def info_values(model: ChoiceModel, draws: Draws, outcome: Outcome, winner: str) -> list[InfoValue]
```

含义：领先者"这一次是否胜出"的波动里，有多大比例可以由这一个输入解释。它回答"弄准它能消除多少结论的不确定"。

固定的规则：
- `y` 是 `outcome.share` 里领先者的那一列；`total = y.var()`。`total == 0` 时返回空列表。
- 参与计算的输入：`is_uncertain` 的数值（分位取 `draws.quantile`）；以及每条权重（分位取 `draws.weight_quantile`），前提是 `model.weights.sigma() > 0` 且准则多于一条。
- 对每个输入：按分位分成 10 个等宽的组（`min(floor(u * 10), 9)`）；`share` 是各组均值相对总均值的加权方差除以 `total`；`leader_win_rate_low` 是第一组里 `y` 的均值，`leader_win_rate_high` 是最后一组的。
- 按 `share` 从大到小排序，相等时保持原有顺序。

- [ ] **Step 1: 写失败的测试**

```python
import pytest
from quanheng.choice.model import ChoiceModel
from quanheng.choice.solve import solve_choice

def test_decisive_and_irrelevant_inputs():
    m = ChoiceModel.model_validate({
        "title": "t", "options": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
        "criteria": [{"id": "x", "name": "X", "bad": 0, "good": 10},
                     {"id": "y", "name": "Y", "bad": 0, "good": 10}],
        "scores": {"a": {"x": 5, "y": [0, 10]}, "b": {"x": [0, 10], "y": [0, 10]}},
        "weights": {"values": {"x": 1, "y": 0}, "certainty": "fixed"}})
    r = solve_choice(m)
    iv = {v.target: v for v in r.info_values}
    assert r.verdict.winner == "a" and r.info_values[0].target == "scores.b.x"
    assert iv["scores.b.x"].share == pytest.approx(1.0, abs=1e-6)
    assert iv["scores.b.x"].leader_win_rate_low == 1.0 and iv["scores.b.x"].leader_win_rate_high == 0.0
    assert iv["scores.a.y"].share < 0.01 and iv["scores.b.y"].share < 0.01
    assert not any(t.startswith("weights.") for t in iv)             # 权重固定，不参与

def test_nothing_to_learn_when_the_winner_never_loses(hand_model):
    assert solve_choice(hand_model).info_values == []

def test_example_ranks_the_guessed_commute_first(rent_model):
    ivs = solve_choice(rent_model).info_values
    first = ivs[0]
    assert first.target == "scores.b.commute" and 0.10 < first.share < 0.30
    assert first.leader_win_rate_low == pytest.approx(0.49, abs=0.05)
    assert first.leader_win_rate_high == pytest.approx(0.98, abs=0.02)
    assert "weights.commute" in {v.target for v in ivs}
    assert [v.share for v in ivs] == sorted((v.share for v in ivs), reverse=True)
```

- [ ] **Step 2: 运行，确认失败**

Run: `uv run pytest tests/test_choice_info_value.py -v`
Expected: 失败，`quanheng.choice.info_value` 不存在。

- [ ] **Step 3: 按固定规则实现，并在 `solve_choice` 里调用**

- [ ] **Step 4: 运行，确认通过**

Run: `uv run pytest tests/test_choice_info_value.py tests/test_choice_solve.py -v && uv run mypy`
Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add src/quanheng/choice/info_value.py src/quanheng/choice/solve.py tests/test_choice_info_value.py
git commit -m "feat: rank uncertain inputs by how much of the outcome they explain"
```

---

### Task 10: 假设追问

**Files:**
- Create: `src/quanheng/whatif.py`
- Test: `tests/test_whatif.py`

**Interfaces:**
- Consumes: `ChoiceModel`、`solve_choice`、`ChoiceResult`、`Thresholds`、`PatchError`
- Produces:

```python
class PatchOp(Model):
    target: str                              # 见下
    op: Literal["set", "scale", "shift"]     # 设为、乘以、加上
    value: float
class Change(Model):
    target: str
    before: list[float]; after: list[float]  # 1、2 或 3 个数，对应确定值、范围、三点估计
class WhatIfResult(Model):
    changes: list[Change]
    before: ChoiceResult; after: ChoiceResult
    win_rate_delta: dict[str, float]         # after 减 before
    winner_changed: bool; level_changed: bool

def apply_patch(model: ChoiceModel, ops: list[PatchOp]) -> tuple[ChoiceModel, list[Change]]
def compare(model: ChoiceModel, ops: list[PatchOp], *, samples: int = 10_000, seed: int = 0,
            thresholds: Thresholds = Thresholds()) -> WhatIfResult
```

固定的规则：
- 允许的目标：`scores.{候选 id}.{准则 id}`、`weights.{准则 id}`、`constraints.{序号}.threshold`。其他一律 `PatchError`；以 `criteria.` 开头的目标，错误信息用 `zh.PATCH_ANCHOR`（"锚点不是假设追问的目标"）。
- 数值目标：`set` 把它变成确定值，来源改为 `user_input`；`scale` 和 `shift` 作用在它的每一个数上，来源和确认状态不变；`scale` 的系数必须大于 0。
- 权重目标：先归一化，对该条权重做运算得到 `v`，要求 `0 <= v <= 1`；其余各条乘以 `(1 - v) / (1 - 原值)`；只有一条准则或原值为 1 时 `PatchError`。结果存为归一化后的值。
- 约束目标：直接对门槛做运算。
- 修改按列表顺序依次执行；不改动传入的模型。
- `compare` 用相同的 `samples`、`seed`、`thresholds` 分别求解修改前后的模型。

- [ ] **Step 1: 写失败的测试**

```python
import numpy as np, pytest
from quanheng.choice.evaluate import draw
from quanheng.errors import PatchError
from quanheng.whatif import PatchOp, apply_patch, compare

def op(target, kind, value):
    return PatchOp(target=target, op=kind, value=value)

def test_scale_shift_set_on_quantities(rent_model):
    m, changes = apply_patch(rent_model, [
        op("scores.a.rent", "scale", 1.2), op("scores.a.commute", "shift", 5),
        op("scores.b.commute", "set", 45), op("constraints.0.threshold", "shift", -10000)])
    assert [c.model_dump() for c in changes] == [
        {"target": "scores.a.rent", "before": [78000.0], "after": [93600.0]},
        {"target": "scores.a.commute", "before": [25.0, 35.0], "after": [30.0, 40.0]},
        {"target": "scores.b.commute", "before": [40.0, 55.0], "after": [45.0]},
        {"target": "constraints.0.threshold", "before": [85000.0], "after": [75000.0]}]
    assert m.scores["b"]["commute"].source.type == "user_input"
    assert m.scores["a"]["commute"].source == rent_model.scores["a"]["commute"].source
    assert rent_model.scores["a"]["rent"].central == 78000          # 原模型没被改动

def test_weight_patch_rescales_the_others(hand_model):
    m, changes = apply_patch(hand_model, [op("weights.rent", "set", 0.8)])
    assert m.weights.normalized() == pytest.approx({"rent": 0.8, "commute": 0.2})
    assert changes[0].before == [0.5] and changes[0].after == [0.8]

def test_compare_reports_a_flip(hand_model):
    r = compare(hand_model, [op("weights.rent", "set", 0.8)])       # 翻盘点在 2/3
    assert r.before.verdict.winner == "a" and r.after.verdict.winner == "b"
    assert r.winner_changed and r.win_rate_delta == {"a": -1.0, "b": 1.0}
    assert r.level_changed is False

def test_rent_increase_breaks_the_budget(rent_model):
    r = compare(rent_model, [op("scores.a.rent", "scale", 1.2)])    # 93,600 超过 85,000 的约束
    assert r.after.feasibility["a"] == 0.0 and r.after.win_rates["b"] == 1.0
    assert r.winner_changed and r.level_changed

def test_unpatched_inputs_draw_the_same_values(rent_model):
    after, _ = apply_patch(rent_model, [op("scores.a.commute", "scale", 1.5)])
    d0, d1 = draw(rent_model, 1000, 0), draw(after, 1000, 0)
    assert np.array_equal(d0.quantile, d1.quantile)                 # 被改的输入仍在相同的分位上
    assert np.array_equal(d0.raw[:, 1, :], d1.raw[:, 1, :])

@pytest.mark.parametrize("target, kind, value", [
    ("scores.z.rent", "set", 1), ("scores.a", "set", 1), ("nonsense", "set", 1),
    ("constraints.5.threshold", "set", 1), ("scores.a.rent", "scale", 0),
    ("scores.a.rent", "scale", -1), ("weights.rent", "set", 1.5), ("weights.rent", "shift", -0.9),
])
def test_invalid_patches(rent_model, target, kind, value):
    with pytest.raises(PatchError):
        apply_patch(rent_model, [op(target, kind, value)])

def test_anchors_are_not_a_target(rent_model):
    with pytest.raises(PatchError, match="锚点"):
        apply_patch(rent_model, [op("criteria.rent.bad", "set", 100000)])
```

- [ ] **Step 2: 运行，确认失败**

Run: `uv run pytest tests/test_whatif.py -v`
Expected: 失败，`quanheng.whatif` 不存在。

- [ ] **Step 3: 按固定规则实现**

本任务需要 `zh.PATCH_ANCHOR`，所以先建 `src/quanheng/zh.py`，只放这一条；任务 11 再补全。

- [ ] **Step 4: 运行，确认通过**

Run: `uv run pytest tests/test_whatif.py -v && uv run mypy`
Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add src/quanheng/whatif.py src/quanheng/zh.py tests/test_whatif.py
git commit -m "feat: what-if patches with before/after comparison on shared random streams"
```

---

### Task 11: 模板讲解

**Files:**
- Create: `src/quanheng/choice/explain.py`
- Modify: `src/quanheng/zh.py`
- Test: `tests/test_explain.py`

**Interfaces:**
- Consumes: `ChoiceModel`、`ChoiceResult`、`WhatIfResult`
- Produces:

```python
def fmt_number(x: float) -> str
def explain_choice(model: ChoiceModel, result: ChoiceResult) -> str
def explain_whatif(model: ChoiceModel, whatif: WhatIfResult) -> str
```

**数字的写法**
- `fmt_number`：先取 4 位有效数字 `r = float(f"{x:.4g}")`；`|r| >= 1000` 写成带千分位的整数；`r` 是整数写成整数；否则写成 `f"{r:g}"`。例：`39.99999 → "40"`，`41.1094 → "41.11"`，`83469.5 → "83,470"`。
- 百分比：胜出率和各种比例写成整数百分比 `f"{p * 100:.0f}%"`；误差写成 `f"±{1.96 * se * 100:.1f}%"`。
- 权重和贡献保留两位小数；贡献带正负号。
- 使用者所填的值：确定值写一个数；范围写成 `40–55`；三点估计写成 `4–7，最可能 5`。

**文案**（全部放在 `zh.py`，花括号是占位）

| 场合 | 文字 |
|---|---|
| 第一行 | `{title}` |
| 结论，明确或倾向 | `{档位} {领先者}　{领先者} 在 {p} 的情况下胜出（{误差}）` |
| 结论，难分高下 | `难分高下　{领先者} 略占上风，在 {p} 的情况下胜出（{误差}）` |
| 结论，多数不可行 | `多数情况下没有可行方案　{p} 的情况下没有任何候选满足全部硬约束` |
| 提醒，紧跟结论 | `在你给出的范围和偏好之内成立，不代表事后一定更好。` |
| 有不可行的情况但不占多数 | `有 {p} 的情况下没有任何候选可行。` |
| 某候选可行比例低于 100% | `{候选} 有 {p} 的情况不满足硬约束。` |
| 贡献的标题 | `为什么是 {领先者}（按中心值计算，{领先者} 相对 {第二名} 的各项贡献）` |
| 贡献的每一行 | `  {准则} {差值}`，按差值的绝对值从大到小 |
| 中心值与抽样不一致 | `注意：按中心值计算是 {候选} 领先，与抽样结果不一致，以抽样结果为准。` |
| 翻盘点的标题 | `什么会让结论翻转` |
| 数值的翻盘点 | `  · {候选} 的"{准则}"{低于或高于} {值} {单位}（你填的是 {所填}）` |
| 权重的翻盘点 | `  · "{准则}"的权重{低于或高于} {值}（现在是 {现值}）` |
| 其余不会翻转的 | `  · 其余 {n} 个输入在参照范围内怎么变都不会翻转结论` |
| 一个翻盘点都没有 | `  · 在参照范围内，任何单个输入怎么变都不会翻转结论` |
| 信息的标题 | `最值得补的信息` |
| 数值的信息 | `  {序号}. {候选} 的"{准则}"：在你给的范围低端时 {领先者} 的胜出率是 {p}，高端时是 {p}` |
| 权重的信息 | `  {序号}. "{准则}"有多重要：权重偏低时 {领先者} 的胜出率是 {p}，偏高时是 {p}` |
| 没有值得补的 | `  没有哪一个输入单独弄准后会明显改变把握。` |
| 数值超出锚点 | `注意：{候选} 的"{准则}"超出了你设的参照范围。` |
| 一致性差 | `注意：两两比较的结果前后不够一致，权重的波动已放大。` |
| 约束差距的标题 | `放宽哪条约束最容易得到可行方案` |
| 约束差距的每一行 | `  · "{准则}" {符号} {门槛}：最接近的是 {候选}，还差 {差距} {单位}` |
| 假设的标题 | `这次计算的假设` |
| 末行 | `抽样 {次数} 次，种子 {种子}，引擎 {版本}，NumPy {版本}` |

假设的五条文字，按顺序：`各个不确定的输入相互独立`、`数值按"差"和"好"两个参照值线性换算成分数`、`只给了范围的输入，范围内每个值同样可能`、`给了最可能值的输入，越靠近最可能值越可能`、`权重按你选的把握程度上下浮动`。每条前面加 `  · `。

**编排规则**
- 档位的中文是"明确""倾向""难分高下""多数情况下没有可行方案"。候选和准则一律用 `name`，不用 `id`。
- "低于"还是"高于"：`flips_at < current` 用"低于"，否则用"高于"。
- 翻盘点最多列 5 条有临界值的，顺序沿用结果里的顺序。
- 信息最多列 3 条，只列 `share >= 0.01` 的。
- 各小节之间空一行。多数不可行时，不输出贡献、翻盘点、信息三节，改为输出约束差距一节。
- `explain_whatif`：每条修改一行 `{目标的中文说法}：{之前} → {之后}`，其中数值写成 `{候选} 的"{准则}"`，权重写成 `"{准则}"的权重`（两位小数），约束写成 `"{准则}"的门槛`；然后是 `之前：{结论行}` 和 `之后：{结论行}`；最后一行是 `结论翻转了。` 或 `结论没有翻转。`（看 `winner_changed`）。

- [ ] **Step 1: 写失败的测试**

```python
import pytest
from quanheng.choice.explain import explain_choice, explain_whatif, fmt_number
from quanheng.choice.model import ChoiceModel
from quanheng.choice.solve import solve_choice
from quanheng.whatif import PatchOp, compare

@pytest.mark.parametrize("x, text", [
    (39.99999, "40"), (41.1094, "41.11"), (83469.5, "83,470"), (78000, "78,000"),
    (0.5, "0.5"), (-1234.4, "-1,234"), (0, "0"),
])
def test_fmt_number(x, text):
    assert fmt_number(x) == text

def test_hand_model_text(hand_model):
    lines = explain_choice(hand_model, solve_choice(hand_model)).splitlines()
    assert lines[0] == "hand"
    assert "明确 A　A 在 100% 的情况下胜出（±0.0%）" in lines
    assert "在你给出的范围和偏好之内成立，不代表事后一定更好。" in lines
    assert "为什么是 A（按中心值计算，A 相对 B 的各项贡献）" in lines
    assert "  通勤 +0.25" in lines
    assert '  · B 的"通勤"低于 40 分钟（你填的是 50）' in lines
    assert '  · "房租"的权重高于 0.67（现在是 0.50）' in lines
    assert "  · 其余 1 个输入在参照范围内怎么变都不会翻转结论" in lines
    assert "  没有哪一个输入单独弄准后会明显改变把握。" in lines
    assert "  · 各个不确定的输入相互独立" in lines
    assert lines[-1].startswith("抽样 10,000 次，种子 0，引擎 0.1.0，NumPy ")

def test_example_text(rent_model):
    text = explain_choice(rent_model, solve_choice(rent_model))
    assert "倾向 A　A 在 81% 的情况下胜出（±0.8%）" in text
    assert '  · B 的"通勤"低于 41.11 分钟（你填的是 40–55）' in text
    assert '  1. B 的"通勤"：在你给的范围低端时 A 的胜出率是' in text
    assert "rent" not in text and "commute" not in text             # 不出现内部标识

def test_mostly_infeasible_text(hand_model):
    data = hand_model.model_dump()
    data["constraints"] = [{"criterion": "rent", "op": "<=", "threshold": 50000}]
    m = ChoiceModel.model_validate(data)
    text = explain_choice(m, solve_choice(m))
    assert "多数情况下没有可行方案　100% 的情况下没有任何候选满足全部硬约束" in text
    assert '  · "房租" <= 50,000：最接近的是 B，还差 10,000 日元/月' in text
    assert "什么会让结论翻转" not in text and "为什么是" not in text

def test_whatif_text(hand_model):
    r = compare(hand_model, [PatchOp(target="weights.rent", op="set", value=0.8)])
    lines = explain_whatif(hand_model, r).splitlines()
    assert lines[0] == '"房租"的权重：0.50 → 0.80'
    assert lines[1] == "之前：明确 A　A 在 100% 的情况下胜出（±0.0%）"
    assert lines[2] == "之后：明确 B　B 在 100% 的情况下胜出（±0.0%）"
    assert lines[-1] == "结论翻转了。"
```

- [ ] **Step 2: 运行，确认失败**

Run: `uv run pytest tests/test_explain.py -v`
Expected: 失败，`quanheng.choice.explain` 不存在。

- [ ] **Step 3: 把文案表写进 `zh.py`，按编排规则实现 `explain.py`**

`explain.py` 里不出现任何中文字面量，全部从 `zh.py` 取。

- [ ] **Step 4: 运行，确认通过**

Run: `uv run pytest tests/test_explain.py -v && uv run mypy`
Expected: 全部通过。

- [ ] **Step 5: 提交**

```bash
git add src/quanheng/choice/explain.py src/quanheng/zh.py tests/test_explain.py
git commit -m "feat: deterministic Chinese explanation of results and what-if comparisons"
```

---

### Task 12: 决策文件、命令行、基准结果

**Files:**
- Create: `src/quanheng/io.py`, `src/quanheng/cli.py`, `src/quanheng/__main__.py`, `examples/golden/rent-a-or-b.result.json`
- Modify: `pyproject.toml`（加命令入口）, `tests/conftest.py`（加 `--update-golden` 选项）, `README.md`, `README.zh-CN.md`, `src/quanheng/zh.py`
- Test: `tests/test_cli.py`, `tests/test_golden.py`

**Interfaces:**
- Consumes: 前面全部
- Produces:

```python
# io.py
def load_model(path: Path) -> ChoiceModel
def save_model(model: ChoiceModel, path: Path) -> None
# cli.py
def main(argv: list[str] | None = None) -> int
```

命令：

```
quanheng solve FILE [--samples N] [--seed S] [--json]
quanheng whatif FILE [--set TARGET=VALUE]... [--scale TARGET=FACTOR]... [--shift TARGET=DELTA]...
                     [--samples N] [--seed S] [--json]
quanheng --version
```

固定的规则：
- `pyproject.toml` 加 `[project.scripts]`：`quanheng = "quanheng.cli:main"`。`python -m quanheng` 同样可用。
- `load_model` 用 `utf-8-sig` 读文件（容忍 BOM）。文件不存在、为空、不是合法 JSON、`kind` 不是 `"choice"`、校验失败，一律抛 `ModelFileError`。校验失败的信息每个错误一行，格式是 `{路径}：{说明}`，路径用点号连接，例如 `scores.b`。
- `save_model` 写 UTF-8、不转义中文、缩进 2 格、末尾换行。
- `main` 一开始把标准输出和标准错误重新配置为 UTF-8、`errors="replace"`。
- 默认输出是 `explain_choice` 或 `explain_whatif` 的文字；`--json` 输出结果的 `model_dump_json(indent=2)`。
- `whatif` 的三种修改按命令行上出现的先后顺序执行。
- 退出码：0 成功；2 文件、模型、参数或修改有错；3 有未确认的数值。出错时信息写到标准错误，不出现调用栈。未确认时逐行列出目标。
- 错误信息的中文放进 `zh.py`。
- 基准结果的比较：把结果里的所有浮点数取 6 位有效数字，去掉 `meta.engine_version` 和 `meta.numpy_version`，再和基准文件比较。`uv run pytest tests/test_golden.py --update-golden` 重新生成基准文件。

- [ ] **Step 1: 写失败的测试**

`tests/test_cli.py`（用 `subprocess` 运行 `sys.executable -m quanheng`，`run(*args, env=None)` 返回 `CompletedProcess`，输出按 UTF-8 解码）：

```python
EXAMPLE = "examples/rent-a-or-b.json"

def test_solve_prints_the_explanation():
    p = run("solve", EXAMPLE)
    assert p.returncode == 0 and p.stdout.splitlines()[0] == "租 A 还是租 B？"
    assert "倾向 A　A 在 81% 的情况下胜出（±0.8%）" in p.stdout and p.stderr == ""

def test_solve_json_and_reproducibility():
    a, b = run("solve", EXAMPLE, "--json"), run("solve", EXAMPLE, "--json")
    data = json.loads(a.stdout)
    assert data["kind"] == "choice_result" and data["meta"]["seed"] == 0
    assert a.stdout == b.stdout
    assert json.loads(run("solve", EXAMPLE, "--json", "--seed", "1").stdout)["win_rates"] != data["win_rates"]

def test_whatif_applies_ops_in_command_line_order():
    p = run("whatif", EXAMPLE, "--scale", "scores.a.rent=1.2")
    assert p.returncode == 0 and p.stdout.splitlines()[-1] == "结论翻转了。"
    p = run("whatif", EXAMPLE, "--scale", "scores.a.rent=2", "--set", "scores.a.rent=78000", "--json")
    assert json.loads(p.stdout)["changes"][-1]["after"] == [78000.0]

def test_version():
    assert run("--version").stdout.strip() == "quanheng 0.1.0"

def test_console_that_cannot_show_chinese():
    p = run("solve", EXAMPLE, env={**os.environ, "PYTHONIOENCODING": "ascii"})
    assert p.returncode == 0 and "倾向 A" in p.stdout

def test_file_with_bom_loads(tmp_path):
    f = tmp_path / "bom.json"
    f.write_text(Path(EXAMPLE).read_text(encoding="utf-8"), encoding="utf-8-sig")
    assert run("solve", str(f)).returncode == 0

@pytest.mark.parametrize("content", ["", "{not json", "[]", '{"kind": "timing"}'])
def test_broken_files(tmp_path, content):
    f = tmp_path / "broken.json"; f.write_text(content, encoding="utf-8")
    p = run("solve", str(f))
    assert p.returncode == 2 and p.stderr.strip() and "Traceback" not in p.stderr

def test_missing_file_and_bad_arguments():
    for args in (["solve", "no-such-file.json"], ["solve", EXAMPLE, "--samples", "50"],
                 ["solve", EXAMPLE, "--samples", "10000000"],
                 ["whatif", EXAMPLE, "--set", "criteria.rent.bad=1"],
                 ["whatif", EXAMPLE, "--set", "scores.a.rent"]):
        p = run(*args)
        assert p.returncode == 2 and "Traceback" not in p.stderr, args

@pytest.mark.parametrize("fn, place", [
    (lambda d: d["scores"]["a"].update(rent=[5, 5]), "scores.a.rent"),   # 字段本身不合法
    (lambda d: d["scores"]["b"].pop("size"), "scores.b"),                # 跨字段：缺一个数值
])
def test_validation_errors_name_the_place(tmp_path, fn, place):
    data = json.loads(Path(EXAMPLE).read_text(encoding="utf-8"))
    fn(data)
    f = tmp_path / "bad.json"; f.write_text(json.dumps(data), encoding="utf-8")
    p = run("solve", str(f))
    assert p.returncode == 2 and place in p.stderr and "Traceback" not in p.stderr

def test_nan_is_rejected(tmp_path):
    text = Path(EXAMPLE).read_text(encoding="utf-8").replace('"rent": 78000', '"rent": NaN')
    f = tmp_path / "nan.json"; f.write_text(text, encoding="utf-8")
    assert run("solve", str(f)).returncode == 2

def test_unconfirmed_inputs_exit_3(tmp_path):
    text = Path(EXAMPLE).read_text(encoding="utf-8").replace('"confirmed": true', '"confirmed": false')
    f = tmp_path / "u.json"; f.write_text(text, encoding="utf-8")
    p = run("solve", str(f))
    assert p.returncode == 3 and "scores.b.commute" in p.stderr
```

`tests/test_golden.py`（`rounded_view` 是这个文件里的辅助函数，递归地把浮点数取 6 位有效数字并去掉两个版本号）：

```python
def test_example_matches_golden(rent_model, request):
    view = rounded_view(solve_choice(rent_model).model_dump())       # 6 位有效数字，去掉两个版本号
    path = Path("examples/golden/rent-a-or-b.result.json")
    if request.config.getoption("--update-golden"):
        path.write_text(json.dumps(view, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert view == golden
    assert golden["verdict"]["level"] == "leaning" and golden["win_rates"] == {"a": 0.8092, "b": 0.1908}

def test_save_and_load_round_trip(rent_model, tmp_path):
    save_model(rent_model, tmp_path / "m.json")
    assert load_model(tmp_path / "m.json") == rent_model
    assert "租 A 还是租 B？" in (tmp_path / "m.json").read_text(encoding="utf-8")
```

`win_rates` 的 `0.8092` 来自对本计划所定算法的事先验证。实现得出的数字不同，说明随机流的标识、权重波动或并列规则与计划不一致，要回头查，而不是改这个数。

- [ ] **Step 2: 运行，确认失败**

Run: `uv run pytest tests/test_cli.py tests/test_golden.py -v`
Expected: 失败，`quanheng.io`、`quanheng.cli` 不存在。

- [ ] **Step 3: 实现 `io.py`、`cli.py`、`__main__.py`，更新 `pyproject.toml` 后运行 `uv sync`**

用标准库的 `argparse`。`--set`、`--scale`、`--shift` 用同一个自定义 action 追加到同一个列表，以保留先后顺序。

- [ ] **Step 4: 生成基准文件**

Run: `uv run pytest tests/test_golden.py --update-golden`
Expected: 通过，并生成 `examples/golden/rent-a-or-b.result.json`。再不带选项运行一次确认通过。

- [ ] **Step 5: 更新两份 README**

各加一节"试一试"：`uv run quanheng solve examples/rent-a-or-b.json` 以及它的真实输出（从终端原样复制），再加一个 `whatif` 的例子。把里程碑表里的 0 和 1a 标为完成。

- [ ] **Step 6: 运行全部检查**

Run: `uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: 全部通过。

- [ ] **Step 7: 提交并推送，确认三个操作系统上都通过**

```bash
git add -A
git commit -m "feat: decision file IO, command line, golden result and quick start"
git push origin main
gh run watch --exit-status
```

Expected: 六个矩阵任务全部成功。基准结果的测试在三个操作系统上都通过，就验证了"跨平台显示一致"。
