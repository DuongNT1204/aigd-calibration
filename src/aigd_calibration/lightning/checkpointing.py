"""Checkpoint loading helpers for CLI commands."""

from __future__ import annotations

from pathlib import Path


def find_checkpoint(run_dir: str | Path, checkpoint: str | Path | None = None) -> Path:
    """Find an explicit checkpoint or the latest checkpoint inside a run."""
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    checkpoint_dir = Path(run_dir) / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")
    return candidates[-1]
