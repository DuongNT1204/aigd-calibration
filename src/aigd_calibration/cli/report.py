"""CLI command for building summary tables."""

from __future__ import annotations

import argparse
import logging


def add_parser(subparsers) -> None:
    """Register `aigd report`."""
    parser = subparsers.add_parser("report", help="Collect metrics from run directories")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Write a CSV comparison table."""
    from aigd_calibration.evaluation.report import write_summary

    write_summary(args.runs, args.output)
    logging.info("Wrote report to %s", args.output)
