"""Main `aigd` CLI entry point."""

from __future__ import annotations

import argparse

from aigd_calibration.utils.logging import configure_logging

from . import calibrate, eval, report, split, train


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="aigd", description="AIGD calibration experiment CLI")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    split.add_parser(subparsers)
    train.add_parser(subparsers)
    eval.add_parser(subparsers)
    calibrate.add_parser(subparsers)
    report.add_parser(subparsers)
    return parser


def main() -> None:
    """CLI entry point installed by pyproject.toml."""
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
