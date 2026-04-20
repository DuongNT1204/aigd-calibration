"""Metadata normalization for AI-generated image detection datasets.

The project uses metadata JSONL files as the source of truth. Image folders can
live anywhere as long as every row points to the image and records its label and
generator/source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABEL_MAP = {
    "0": 0,
    "real": 0,
    "negative": 0,
    "1": 1,
    "fake": 1,
    "synthetic": 1,
    "semisynthetic": 1,
    "ai": 1,
    "generated": 1,
}


@dataclass(frozen=True)
class ImageRecord:
    """Normalized representation of one image sample."""

    image_path: Path
    label: int
    generator: str
    source: str
    raw: dict[str, Any]


def normalize_label(value: Any) -> int:
    """Map common real/fake label strings to 0/1."""
    key = str(value).strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"Unsupported label {value!r}. Use one of {sorted(LABEL_MAP)}")
    return LABEL_MAP[key]


def parse_record(record: dict[str, Any], base_dir: str | Path | None = None) -> ImageRecord:
    """Normalize one metadata row into an ImageRecord."""
    image_value = record.get("image_path") or record.get("path") or record.get("file")
    if not image_value:
        raise ValueError(f"Record is missing image_path/path/file: {record}")

    image_path = Path(str(image_value))
    if not image_path.is_absolute() and base_dir is not None:
        image_path = Path(base_dir) / image_path

    label = normalize_label(record.get("label"))
    generator = str(record.get("generator") or record.get("model") or record.get("dataset") or "real")
    source = str(record.get("source") or record.get("dataset") or generator)
    return ImageRecord(image_path=image_path, label=label, generator=generator, source=source, raw=record)


def to_jsonl_row(record: dict[str, Any]) -> dict[str, Any]:
    """Return a clean JSONL row for split files."""
    row = dict(record)
    row["label"] = normalize_label(row.get("label"))
    row["generator"] = str(row.get("generator") or row.get("model") or row.get("dataset") or "real")
    row["image_path"] = row.get("image_path") or row.get("path") or row.get("file")
    if not row["image_path"]:
        raise ValueError(f"Record is missing image path: {record}")
    return row
