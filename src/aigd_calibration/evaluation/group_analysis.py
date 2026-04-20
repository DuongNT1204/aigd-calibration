"""Metric breakdowns by generator/source."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .metrics import compute_binary_metrics


def metrics_by_group(records: list[dict], group_key: str = "generator", n_bins: int = 15) -> list[dict]:
    """Compute metrics per generator or source."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[str(row.get(group_key, "unknown"))].append(row)

    rows = []
    for group, group_records in sorted(groups.items()):
        labels = np.asarray([row["label"] for row in group_records], dtype=np.int64)
        probs = np.asarray([row["prob_fake"] for row in group_records], dtype=np.float64)
        metrics = compute_binary_metrics(labels, probs, n_bins=n_bins)
        metrics[group_key] = group
        rows.append(metrics)
    return rows
