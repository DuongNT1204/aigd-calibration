"""PyTorch Lightning Trainer builder."""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl

from .callbacks import build_callbacks
from .loggers import build_logger


def build_trainer(config: dict, run_dir: str | Path, enable_checkpointing: bool = True) -> pl.Trainer:
    """Construct a Lightning Trainer from config."""
    training_cfg = config.get("training", {})
    callbacks = build_callbacks(config, Path(run_dir) / "checkpoints") if enable_checkpointing else []
    logger = build_logger(config, run_dir)

    lightning_strategy = training_cfg.get("lightning_strategy", "auto")
    trainer_kwargs = {
        "max_epochs": int(training_cfg.get("epochs", 10)),
        "accelerator": training_cfg.get("accelerator", "auto"),
        "devices": training_cfg.get("devices", "auto"),
        "precision": training_cfg.get("precision", "32-true"),
        "callbacks": callbacks,
        "logger": logger,
        "log_every_n_steps": int(training_cfg.get("log_every_n_steps", 20)),
        "default_root_dir": str(run_dir),
        "sync_batchnorm": bool(training_cfg.get("sync_batchnorm", False)),
        "use_distributed_sampler": bool(training_cfg.get("use_distributed_sampler", True)),
        "accumulate_grad_batches": int(training_cfg.get("accumulate_grad_batches", 1)),
        "gradient_clip_val": float(training_cfg.get("gradient_clip_val", 0.0)),
        "deterministic": bool(training_cfg.get("deterministic", False)),
        "benchmark": bool(training_cfg.get("benchmark", False)),
        "num_sanity_val_steps": int(training_cfg.get("num_sanity_val_steps", 2)),
        "fast_dev_run": bool(training_cfg.get("fast_dev_run", False)),
        "limit_train_batches": training_cfg.get("limit_train_batches", 1.0),
        "limit_val_batches": training_cfg.get("limit_val_batches", 1.0),
        "limit_test_batches": training_cfg.get("limit_test_batches", 1.0),
        "limit_predict_batches": training_cfg.get("limit_predict_batches", 1.0),
    }

    if lightning_strategy not in (None, "", "auto"):
        trainer_kwargs["strategy"] = lightning_strategy

    return pl.Trainer(
        **trainer_kwargs,
    )
