"""Helpers for writing comparison tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


_REPORT_COLUMN_ORDER = ["run", "split", "calibration", "accuracy", "mcc", "f1", "ap", "auc", "nll", "brier", "ece", "count"]


def write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    """Write a list of dictionaries to CSV with canonical column ordering."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    all_keys = list(rows[0].keys())
    ordered = [k for k in _REPORT_COLUMN_ORDER if k in all_keys]
    remaining = [k for k in all_keys if k not in ordered]
    fieldnames = ordered + remaining
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
