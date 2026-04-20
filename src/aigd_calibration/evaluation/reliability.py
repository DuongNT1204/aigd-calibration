"""Reliability diagram statistics and plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def reliability_bins(labels: np.ndarray, probs_fake: np.ndarray, n_bins: int = 15) -> dict:
    """Return binned confidence/accuracy stats for a reliability diagram."""
    pred = (probs_fake >= 0.5).astype(np.int64)
    confidence = np.maximum(probs_fake, 1.0 - probs_fake)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    acc = np.zeros(n_bins, dtype=np.float64)
    conf = np.zeros(n_bins, dtype=np.float64)
    frac = np.zeros(n_bins, dtype=np.float64)

    for idx in range(n_bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (confidence > left) & (confidence <= right)
        if idx == 0:
            mask = (confidence >= left) & (confidence <= right)
        if np.any(mask):
            acc[idx] = correct[mask].mean()
            conf[idx] = confidence[mask].mean()
            frac[idx] = mask.mean()
    return {"edges": edges, "centers": centers, "accuracy": acc, "confidence": conf, "fraction": frac}


def plot_reliability(labels: np.ndarray, probs_fake: np.ndarray, output: str | Path, n_bins: int = 15, title: str = "Reliability") -> None:
    """Create a reliability diagram PNG."""
    import matplotlib.pyplot as plt

    stats = reliability_bins(labels, probs_fake, n_bins=n_bins)
    width = 1.0 / n_bins * 0.9
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [1, 2]})
    axes[0].bar(stats["centers"], stats["fraction"], width=width, color="#4C78A8", edgecolor="black")
    axes[0].set_ylabel("Fraction")
    axes[0].set_title(title)

    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].bar(stats["centers"], stats["accuracy"], width=width, color="#4C78A8", edgecolor="black")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
