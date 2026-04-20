"""Saved-logits artifact format.

Post-hoc calibration should operate on validation logits and then apply the
learned calibrator to test logits. This file defines that shared JSONL format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from aigd_calibration.utils.io import read_jsonl, write_jsonl


def logits_to_prob_fake(logits: torch.Tensor | np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert [N,2] logits to fake-class probabilities."""
    tensor = torch.as_tensor(logits, dtype=torch.float32) / float(temperature)
    return torch.softmax(tensor, dim=-1)[:, 1].cpu().numpy()


def make_logit_records(batch: dict[str, Any], logits: torch.Tensor, split: str) -> list[dict[str, Any]]:
    """Build JSONL rows from one prediction batch."""
    logits_cpu = logits.detach().float().cpu()
    probs_fake = logits_to_prob_fake(logits_cpu)
    rows: list[dict[str, Any]] = []

    if isinstance(batch, dict):
        labels = batch["label"].detach().cpu().tolist()
        image_paths = batch.get("image_path", ["unknown"] * len(labels))
        generators = batch.get("generator", ["unknown"] * len(labels))
        sources = batch.get("source", ["unknown"] * len(labels))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        labels = batch[1].detach().cpu().tolist()
        image_paths = ["unknown"] * len(labels)
        generators = ["unknown"] * len(labels)
        sources = ["unknown"] * len(labels)
    else:
        raise TypeError(f"Unsupported batch format: {type(batch)}")

    for idx, label in enumerate(labels):
        rows.append(
            {
                "image_path": image_paths[idx],
                "label": int(label),
                "generator": generators[idx],
                "source": sources[idx],
                "split": split,
                "logits": logits_cpu[idx].tolist(),
                "prob_fake": float(probs_fake[idx]),
            }
        )
    return rows


def save_logit_records(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write logit records to JSONL."""
    write_jsonl(path, records)


def load_logits(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load logits and labels from a saved JSONL artifact."""
    records = read_jsonl(path)
    logits = np.asarray([row["logits"] for row in records], dtype=np.float32)
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    return logits, labels, records


def update_records_with_temperature(records: list[dict[str, Any]], temperature: float) -> list[dict[str, Any]]:
    """Return copies of logit rows with calibrated fake probabilities."""
    logits = np.asarray([row["logits"] for row in records], dtype=np.float32)
    probs_fake = logits_to_prob_fake(logits, temperature=temperature)
    calibrated = []
    for row, prob in zip(records, probs_fake):
        item = dict(row)
        item["temperature"] = float(temperature)
        item["prob_fake"] = float(prob)
        calibrated.append(item)
    return calibrated
