"""Evaluate a saved logits file."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aigd_calibration.artifacts.logits import load_logits, logits_to_prob_fake
from aigd_calibration.utils.io import write_json

from .metrics import compute_binary_metrics


def evaluate_logits_file(
    logits_path: str | Path,
    output_path: str | Path | None = None,
    n_bins: int = 15,
    threshold: float = 0.5,
) -> dict:
    """Compute metrics from a logits JSONL artifact."""
    logits, labels, records = load_logits(logits_path)
    probs_fake = np.asarray([row.get("prob_fake") for row in records], dtype=np.float64)
    if np.any(np.isnan(probs_fake)):
        probs_fake = logits_to_prob_fake(logits)
    metrics = compute_binary_metrics(labels, probs_fake, n_bins=n_bins, threshold=threshold)
    if output_path:
        write_json(output_path, metrics)
    return metrics
