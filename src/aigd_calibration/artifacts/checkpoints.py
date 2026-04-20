"""Checkpoint path helpers."""

from __future__ import annotations

from pathlib import Path


def best_checkpoint_path(run_dir: str | Path) -> Path:
    """Return the expected best checkpoint path for a run."""
    checkpoint_dir = Path(run_dir) / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No Lightning checkpoints found in {checkpoint_dir}")
    # Lightning checkpoint names include monitored metric values; newest is a safe default for base code.
    return candidates[-1]
