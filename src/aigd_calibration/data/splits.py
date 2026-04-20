"""Cross-generator split generation.

This module creates train/val/test_id/test_ood JSONL files while ensuring that
the selected OOD fake generators never appear in the training split.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from aigd_calibration.utils.io import read_records, write_json, write_jsonl

from .metadata import to_jsonl_row


def _take_fraction(records: list[dict], fraction: float) -> tuple[list[dict], list[dict]]:
    count = int(round(len(records) * fraction))
    return records[:count], records[count:]


def _split_id_records(records: list[dict], val_ratio: float, test_ratio: float) -> tuple[list[dict], list[dict], list[dict]]:
    val, rest = _take_fraction(records, val_ratio)
    denom = max(1e-8, 1.0 - val_ratio)
    test, train = _take_fraction(rest, test_ratio / denom)
    return train, val, test


def build_cross_generator_splits(
    records: Iterable[dict],
    ood_generators: set[str],
    seed: int = 42,
    val_ratio: float = 0.1,
    test_id_ratio: float = 0.1,
    ood_real_ratio: float = 0.1,
) -> dict[str, list[dict]]:
    """Build split rows from metadata records."""
    rng = random.Random(seed)
    rows = [to_jsonl_row(record) for record in records]
    rng.shuffle(rows)

    real_rows = [row for row in rows if row["label"] == 0]
    fake_rows = [row for row in rows if row["label"] == 1]

    fake_by_generator: dict[str, list[dict]] = defaultdict(list)
    for row in fake_rows:
        fake_by_generator[row["generator"]].append(row)

    id_fake: list[dict] = []
    ood_fake: list[dict] = []
    for generator, group in fake_by_generator.items():
        if generator in ood_generators:
            ood_fake.extend(group)
        else:
            id_fake.extend(group)

    if ood_generators and not ood_fake:
        raise ValueError(f"No fake samples found for OOD generators: {sorted(ood_generators)}")

    train_real, val_real, test_id_real = _split_id_records(real_rows, val_ratio, test_id_ratio)
    ood_real, train_real = _take_fraction(train_real, ood_real_ratio)
    train_fake, val_fake, test_id_fake = _split_id_records(id_fake, val_ratio, test_id_ratio)

    splits = {
        "train": train_real + train_fake,
        "val": val_real + val_fake,
        "test_id": test_id_real + test_id_fake,
        "test_ood": ood_real + ood_fake,
    }
    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


def write_cross_generator_splits(
    metadata_path: str | Path,
    out_dir: str | Path,
    ood_generators: set[str],
    seed: int = 42,
    val_ratio: float = 0.1,
    test_id_ratio: float = 0.1,
    ood_real_ratio: float = 0.1,
) -> dict:
    """Read metadata, build splits, write JSONL files, and return a summary."""
    records = read_records(metadata_path)
    splits = build_cross_generator_splits(records, ood_generators, seed, val_ratio, test_id_ratio, ood_real_ratio)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)

    summary = {
        "metadata": str(metadata_path),
        "ood_generators": sorted(ood_generators),
        "counts": {name: len(rows) for name, rows in splits.items()},
    }
    write_json(out_dir / "summary.json", summary)
    return summary
