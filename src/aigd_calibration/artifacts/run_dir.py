"""Run directory layout helpers.

Each experiment writes the same subfolders so CLI commands and report scripts
can find checkpoints, logits, metrics, figures, and tables predictably.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RunDir:
    """Paths for one experiment output directory."""

    root: Path
    checkpoints: Path
    logits: Path
    metrics: Path
    figures: Path
    tables: Path


def create_run_dir(output_root: str | Path, experiment_name: str, config: dict | None = None) -> RunDir:
    """Create the standard output folders for one run."""
    root = Path(output_root) / experiment_name
    run_dir = RunDir(
        root=root,
        checkpoints=root / "checkpoints",
        logits=root / "logits",
        metrics=root / "metrics",
        figures=root / "figures",
        tables=root / "tables",
    )
    for path in (run_dir.root, run_dir.checkpoints, run_dir.logits, run_dir.metrics, run_dir.figures, run_dir.tables):
        path.mkdir(parents=True, exist_ok=True)
    if config is not None:
        with (run_dir.root / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return run_dir
