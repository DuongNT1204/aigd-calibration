"""CLI command for training Lightning models."""

from __future__ import annotations

import argparse
import logging


def add_parser(subparsers) -> None:
    """Register `aigd train`."""
    parser = subparsers.add_parser("train", help="Train a detector experiment")
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Train one configured experiment."""
    from aigd_calibration.artifacts.run_dir import create_run_dir
    from aigd_calibration.config.loader import load_config
    from aigd_calibration.config.schema import validate_train_config
    from aigd_calibration.data.datamodule import AIGDDataModule
    from aigd_calibration.lightning.trainer import build_trainer
    from aigd_calibration.methods.train_time.strategies.registry import build_lightning_module
    from aigd_calibration.utils.seed import set_seed

    config = load_config(args.config)
    validate_train_config(config)
    set_seed(int(config.get("seed", 42)))

    run_dir = create_run_dir(config.get("output_root", "outputs"), config["experiment"]["name"], config)
    datamodule = AIGDDataModule(config)
    module = build_lightning_module(config)
    trainer = build_trainer(config, run_dir.root)
    trainer.fit(module, datamodule=datamodule)
    logging.info("Training finished. Run directory: %s", run_dir.root)
