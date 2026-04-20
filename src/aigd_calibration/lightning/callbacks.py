"""Lightning callback builders."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint


def build_callbacks(config: dict, checkpoint_dir: str | Path) -> list:
    """Build default callbacks for training."""
    training_cfg = config.get("training", {})
    monitor = str(training_cfg.get("checkpoint_metric", "val/auc"))
    mode = str(training_cfg.get("checkpoint_mode", "max"))
    save_top_k = int(training_cfg.get("save_top_k", 1))
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch={epoch:03d}-auc={val/auc:.4f}-ece={val/ece:.4f}",
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=save_top_k > 0,
        auto_insert_metric_name=False,
    )
    return [checkpoint, LearningRateMonitor(logging_interval="epoch")]
