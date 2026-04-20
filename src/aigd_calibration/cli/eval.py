"""CLI command for exporting logits and metrics from a trained run."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def add_parser(subparsers) -> None:
    """Register `aigd eval`."""
    parser = subparsers.add_parser("eval", help="Export logits and compute metrics")
    parser.add_argument("--run", required=True, help="Run directory containing config.yaml and checkpoints")
    parser.add_argument("--split", required=True, choices=["val", "test_id", "test_ood"], help="Split to evaluate")
    parser.add_argument("--checkpoint", help="Optional checkpoint path")
    parser.set_defaults(func=run)


def _load_module(config: dict, checkpoint_path: Path):
    from aigd_calibration.methods.train_time.strategies.diff_dml import DiffDMLModule
    from aigd_calibration.methods.train_time.strategies.standard import StandardAIGDModule

    strategy = str(config.get("training", {}).get("strategy", "standard")).lower()
    cls = DiffDMLModule if strategy in {"diff_dml", "dml"} else StandardAIGDModule
    return cls.load_from_checkpoint(str(checkpoint_path), config=config)


def run(args: argparse.Namespace) -> None:
    """Run prediction, save logits, save metrics, and plot reliability."""
    import numpy as np

    from aigd_calibration.artifacts.logits import make_logit_records, save_logit_records
    from aigd_calibration.config.loader import load_config
    from aigd_calibration.data.datamodule import AIGDDataModule
    from aigd_calibration.evaluation.evaluator import evaluate_logits_file
    from aigd_calibration.evaluation.reliability import plot_reliability
    from aigd_calibration.lightning.checkpointing import find_checkpoint
    from aigd_calibration.lightning.trainer import build_trainer

    run_dir = Path(args.run)
    config = load_config(run_dir / "config.yaml")
    split_path = config["data"][args.split]
    config["data"]["predict"] = split_path

    checkpoint = find_checkpoint(run_dir, args.checkpoint)
    module = _load_module(config, checkpoint)
    datamodule = AIGDDataModule(config)

    # Force single GPU for predict — DDP pads the dataset which causes duplicate predictions.
    eval_config = {**config, "training": {**config.get("training", {}), "devices": 1, "lightning_strategy": "auto"}}
    trainer = build_trainer(eval_config, run_dir, enable_checkpointing=False)

    predictions = trainer.predict(module, datamodule=datamodule)
    records = []
    for item in predictions:
        records.extend(make_logit_records(item["batch"], item["logits"], split=args.split))

    logits_path = run_dir / "logits" / f"{args.split}.jsonl"
    metrics_path = run_dir / "metrics" / f"{args.split}.json"
    figure_path = run_dir / "figures" / f"reliability_{args.split}.png"
    save_logit_records(logits_path, records)
    metrics_cfg = config.get("metrics", {})
    metrics = evaluate_logits_file(
        logits_path,
        metrics_path,
        n_bins=int(metrics_cfg.get("ece_bins", 15)),
        threshold=float(metrics_cfg.get("threshold", 0.5)),
    )

    labels = np.asarray([row["label"] for row in records])
    probs = np.asarray([row["prob_fake"] for row in records])
    plot_reliability(labels, probs, figure_path, n_bins=int(metrics_cfg.get("ece_bins", 15)), title=args.split)
    logging.info("Saved logits to %s", logits_path)
    logging.info("Metrics: %s", metrics)
