"""Command line: ``quanheng solve`` and ``quanheng whatif``."""

import argparse
import sys
from pathlib import Path
from typing import Any, Literal, cast

from quanheng import __version__, zh
from quanheng.choice.evaluate import MAX_SAMPLES, MIN_SAMPLES
from quanheng.choice.explain import explain_choice, explain_whatif
from quanheng.choice.solve import solve_choice
from quanheng.errors import QuanhengError, UnconfirmedInputError
from quanheng.io import load_model
from quanheng.whatif import PatchOp, compare


def _samples(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(zh.SAMPLES_NOT_INT.format(value=value)) from None
    if not MIN_SAMPLES <= n <= MAX_SAMPLES:
        raise argparse.ArgumentTypeError(
            zh.SAMPLES_OUT_OF_RANGE.format(low=MIN_SAMPLES, high=MAX_SAMPLES, value=value)
        )
    return n


def _seed(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(zh.SEED_NOT_INT.format(value=value)) from None


def _patch_value(text: str) -> tuple[str, float]:
    target, sep, number = text.partition("=")
    if not sep or not target:
        raise argparse.ArgumentTypeError(zh.PATCH_MISSING_EQUALS.format(value=text))
    try:
        return target, float(number)
    except ValueError:
        raise argparse.ArgumentTypeError(zh.PATCH_NOT_NUMBER.format(value=text)) from None


class _AppendOp(argparse.Action):
    """Append a PatchOp to one shared list, keeping command-line order."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        target, value = values
        op = cast("Literal['set', 'scale', 'shift']", self.const)
        ops: list[PatchOp] = list(getattr(namespace, self.dest, None) or [])
        ops.append(PatchOp(target=target, op=op, value=value))
        setattr(namespace, self.dest, ops)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quanheng")
    parser.add_argument("--version", action="version", version=f"quanheng {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("file", type=Path, metavar="FILE")
        p.add_argument("--samples", type=_samples, default=10_000, metavar="N")
        p.add_argument("--seed", type=_seed, default=0, metavar="S")
        p.add_argument("--json", action="store_true", dest="as_json")

    common(sub.add_parser("solve"))

    whatif = sub.add_parser("whatif")
    common(whatif)
    for flag, op in (("--set", "set"), ("--scale", "scale"), ("--shift", "shift")):
        whatif.add_argument(
            flag,
            action=_AppendOp,
            const=op,
            dest="ops",
            type=_patch_value,
            metavar="TARGET=VALUE",
        )
    return parser


def _run(args: argparse.Namespace) -> str:
    model = load_model(args.file)
    if args.command == "solve":
        result = solve_choice(model, samples=args.samples, seed=args.seed)
        return result.model_dump_json(indent=2) if args.as_json else explain_choice(model, result)
    ops: list[PatchOp] = args.ops or []
    comparison = compare(model, ops, samples=args.samples, seed=args.seed)
    if args.as_json:
        return comparison.model_dump_json(indent=2)
    return explain_whatif(model, comparison)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    args = _build_parser().parse_args(argv)
    try:
        output = _run(args)
    except UnconfirmedInputError as exc:
        print(zh.UNCONFIRMED_HEADER, file=sys.stderr)
        for target in exc.targets:
            print(target, file=sys.stderr)
        return 3
    except QuanhengError as exc:
        print(f"{zh.ERROR_PREFIX}{exc}", file=sys.stderr)
        return 2
    print(output)
    return 0
