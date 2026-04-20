"""CLI command for post-hoc calibration on saved logits."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def add_parser(subparsers) -> None:
    """Register `aigd calibrate`."""
    parser = subparsers.add_parser("calibrate", help="Fit/apply post-hoc calibration")
    parser.add_argument("--config", required=True, help="Calibration YAML config")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Fit calibrator on validation logits and apply to one or more logits files."""
    import numpy as np

    from aigd_calibration.artifacts.logits import load_logits, save_logit_records
    from aigd_calibration.config.loader import load_config
    from aigd_calibration.evaluation.metrics import compute_binary_metrics
    from aigd_calibration.methods.post_hoc.registry import build_calibrator
    from aigd_calibration.utils.io import write_json

    config = load_config(args.config)
    calibration_cfg = config["calibration"]
    calibrator = build_calibrator(calibration_cfg)

    val_logits, val_labels, _ = load_logits(calibration_cfg["val_logits"])
    calibrator.fit(val_logits, val_labels)

    out_dir = Path(calibration_cfg.get("out_dir", "outputs/calibrated"))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "calibrator.json", calibrator.state_dict())

    apply_logits = calibration_cfg["apply_logits"]
    if isinstance(apply_logits, str):
        apply_logits = [apply_logits]

    for path in apply_logits:
        logits, labels, records = load_logits(path)
        probs = calibrator.predict_proba(logits)
        calibrated_records = []
        for row, prob in zip(records, probs):
            item = dict(row)
            item["prob_fake"] = float(prob)
            item["calibration"] = calibrator.state_dict()
            calibrated_records.append(item)

        stem = Path(path).stem
        output_logits = out_dir / f"{stem}_calibrated.jsonl"
        output_metrics = out_dir / f"{stem}_metrics.json"
        save_logit_records(output_logits, calibrated_records)
        metrics_cfg = config.get("metrics", {})
        metrics = compute_binary_metrics(
            labels,
            np.asarray(probs),
            n_bins=int(metrics_cfg.get("ece_bins", 15)),
            threshold=float(metrics_cfg.get("threshold", 0.5)),
        )
        write_json(output_metrics, metrics)
        logging.info("Calibrated %s -> %s", path, output_logits)
