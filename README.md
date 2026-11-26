# Quanheng 权衡

English | [简体中文](README.zh-CN.md)

Quanheng is a formula-first decision engine. You describe a decision, and the engine compares your options with transparent, reproducible math. The principle: **the LLM only translates; every number and every conclusion is produced by deterministic code.**

## Status

Early development. Milestones (from section 12 of the design document):

| # | Scope | What you can do when it is done | Status |
|---|---|---|---|
| 0 | Repository skeleton: package layout, CI, bilingual README | Install it and run the tests | In progress |
| 1a | Values and provenance, sampling, choice-model engine, CLI | Solve a hand-written decision file | Not started |
| 1b | Service layer and UI, manual form path | Complete a two-option decision in the browser | Not started |
| 1c | LLM layer: drafting, explanation, natural-language follow-up on assumptions | Start a decision from a paragraph of text | Not started |
| 2 | Timing model | Designed separately | Not started |
| 3 | Allocation model | Designed separately | Not started |
| 4 | File import | Designed separately | Not started |
| 5 | Desktop shell | Later | Not started |

## Development

```bash
uv sync
uv run pytest
```

## Design document

See [docs/superpowers/specs/2026-11-26-quanheng-design.md](docs/superpowers/specs/2026-11-26-quanheng-design.md) (written in Chinese).

## License

MIT. See [LICENSE](LICENSE).
