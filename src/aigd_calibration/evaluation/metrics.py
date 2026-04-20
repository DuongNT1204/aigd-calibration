"""Discrimination and calibration metrics.

These metrics are used for both raw model outputs and post-hoc calibrated
outputs, so they accept labels and fake-class probabilities directly.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, matthews_corrcoef, roc_auc_score


def expected_calibration_error(labels: np.ndarray, probs_fake: np.ndarray, n_bins: int = 15) -> float:
    """Compute confidence-based ECE for binary classification."""
    labels = labels.astype(np.int64)
    probs_fake = np.clip(probs_fake.astype(np.float64), 1e-7, 1 - 1e-7)
    pred = (probs_fake >= 0.5).astype(np.int64)
    confidence = np.maximum(probs_fake, 1.0 - probs_fake)
    correct = (pred == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (confidence > left) & (confidence <= right)
        if idx == 0:
            mask = (confidence >= left) & (confidence <= right)
        if not np.any(mask):
            continue
        ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return ece


def compute_binary_metrics(labels: np.ndarray, probs_fake: np.ndarray, n_bins: int = 15, threshold: float = 0.5) -> dict:
    """Compute thesis metrics from labels and fake probabilities."""
    labels = labels.astype(np.int64)
    probs_fake = np.clip(probs_fake.astype(np.float64), 1e-7, 1 - 1e-7)
    pred = (probs_fake >= threshold).astype(np.int64)

    metrics = {
        "accuracy": float(accuracy_score(labels, pred)),
        "mcc": float(matthews_corrcoef(labels, pred)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "ap": float(average_precision_score(labels, probs_fake)) if len(np.unique(labels)) > 1 else None,
        "auc": float(roc_auc_score(labels, probs_fake)) if len(np.unique(labels)) > 1 else None,
        "nll": float(log_loss(labels, np.stack([1.0 - probs_fake, probs_fake], axis=1), labels=[0, 1])),
        "brier": float(np.mean((probs_fake - labels) ** 2)),
        "ece": expected_calibration_error(labels, probs_fake, n_bins=n_bins),
        "count": int(len(labels)),
    }
    return metrics
