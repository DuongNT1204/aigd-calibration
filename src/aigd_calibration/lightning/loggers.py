"""Lightning logger builders."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger


def build_logger(config: dict, run_dir: str | Path):
    """Return both CSVLogger and TensorBoardLogger."""
    experiment_name = config.get("experiment", {}).get("name", "experiment")
    run_dir = Path(run_dir)
    csv = CSVLogger(save_dir=str(run_dir / "logs"), name=str(experiment_name))
    tb = TensorBoardLogger(save_dir=str(run_dir / "tb_logs"), name=str(experiment_name))
    return [csv, tb]
