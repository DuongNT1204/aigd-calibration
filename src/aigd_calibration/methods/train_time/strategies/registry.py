"""Training strategy registry."""

from __future__ import annotations

from .diff_dml import DiffDMLModule
from .standard import StandardAIGDModule


def build_lightning_module(config: dict):
    """Build the correct LightningModule for the configured strategy."""
    strategy = str(config.get("training", {}).get("strategy", "standard")).lower()
    if strategy == "standard":
        return StandardAIGDModule(config)
    if strategy in {"diff_dml", "dml"}:
        return DiffDMLModule(config)
    raise ValueError(f"Unknown training strategy: {strategy}")
