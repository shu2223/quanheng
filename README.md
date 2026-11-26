# Quanheng 权衡

English | [简体中文](README.zh-CN.md)

Quanheng is a formula-first decision engine. You describe a decision, and the engine compares your options with transparent, reproducible math. The principle: **the LLM only translates; every number and every conclusion is produced by deterministic code.**

## Status

Early development. Milestones (from section 12 of the design document):

| # | Scope | What you can do when it is done | Status |
|---|---|---|---|
| 0 | Repository skeleton: package layout, CI, bilingual README | Install it and run the tests | Done |
| 1a | Values and provenance, sampling, choice-model engine, CLI | Solve a hand-written decision file | Done |
| 1b | Service layer and UI, manual form path | Complete a two-option decision in the browser | Not started |
| 1c | LLM layer: drafting, explanation, natural-language follow-up on assumptions | Start a decision from a paragraph of text | Not started |
| 2 | Timing model | Designed separately | Not started |
| 3 | Allocation model | Designed separately | Not started |
| 4 | File import | Designed separately | Not started |
| 5 | Desktop shell | Later | Not started |

## Try it

Solve the bundled example. The interface is Chinese, so the output is too.

```bash
uv run quanheng solve examples/rent-a-or-b.json
```

```text
租 A 还是租 B？
倾向 A　A 在 81% 的情况下胜出（±0.8%）
在你给出的范围和偏好之内成立，不代表事后一定更好。

为什么是 A（按中心值计算，A 相对 B 的各项贡献）
  通勤 +0.15
  房租 -0.08
  面积 -0.04
  采光 +0.03

什么会让结论翻转
  · "通勤"的权重低于 0.25（现在是 0.35）
  · "房租"的权重高于 0.42（现在是 0.30）
  · B 的"通勤"低于 41.11 分钟（你填的是 40–55）
  · A 的"通勤"高于 36.21 分钟（你填的是 25–35）
  · "面积"的权重高于 0.37（现在是 0.20）
  · 其余 1 个输入在参照范围内怎么变都不会翻转结论

最值得补的信息
  1. B 的"通勤"：在你给的范围低端时 A 的胜出率是 49%，高端时是 98%
  2. "通勤"有多重要：权重偏低时 A 的胜出率是 47%，偏高时是 97%
  3. A 的"通勤"：在你给的范围低端时 A 的胜出率是 95%，高端时是 60%

这次计算的假设
  · 各个不确定的输入相互独立
  · 数值按"差"和"好"两个参照值线性换算成分数
  · 只给了范围的输入，范围内每个值同样可能
  · 给了最可能值的输入，越靠近最可能值越可能
  · 权重按你选的把握程度上下浮动

抽样 10,000 次，种子 0，引擎 0.1.0，NumPy 2.4.6
```

Ask what happens if A's rent goes up by 20% (`--set`, `--scale` and `--shift` can be combined and are applied in the order given):

```bash
uv run quanheng whatif examples/rent-a-or-b.json --scale scores.a.rent=1.2
```

```text
A 的"房租"：78,000 → 93,600
之前：倾向 A　A 在 81% 的情况下胜出（±0.8%）
之后：明确 B　B 在 100% 的情况下胜出（±0.0%）
结论翻转了。
```

Add `--json` for machine-readable output, and `--samples N` / `--seed S` to change the sampling. Exit codes: 0 success, 2 bad file or arguments, 3 unconfirmed numbers.

## Development

```bash
uv sync
uv run pytest
```

## Design document

See [docs/superpowers/specs/2026-11-26-quanheng-design.md](docs/superpowers/specs/2026-11-26-quanheng-design.md) (written in Chinese).

## License

MIT. See [LICENSE](LICENSE).
