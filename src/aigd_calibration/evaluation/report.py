"""Report generation across experiment runs."""

from __future__ import annotations

from pathlib import Path

from aigd_calibration.artifacts.tables import write_csv
from aigd_calibration.utils.io import read_json


def collect_metrics(run_dirs: list[str | Path]) -> list[dict]:
    """Collect metric JSON files from multiple run directories."""
    rows = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        for metrics_path in sorted((run_dir / "metrics").glob("*.json")):
            row = read_json(metrics_path)
            row["run"] = run_dir.name
            row["split"] = metrics_path.stem
            row["calibration"] = "none"
            rows.append(row)
        temp_dir = run_dir / "temperature"
        if temp_dir.exists():
            for metrics_path in sorted(temp_dir.glob("*_metrics.json")):
                row = read_json(metrics_path)
                row["run"] = run_dir.name
                row["split"] = metrics_path.stem.replace("_metrics", "")
                row["calibration"] = "temperature"
                rows.append(row)
    return rows


def write_summary(run_dirs: list[str | Path], output_csv: str | Path) -> None:
    """Write a CSV summary for thesis tables."""
    write_csv(output_csv, collect_metrics(run_dirs))
