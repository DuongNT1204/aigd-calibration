"""Model registry.

Training strategies call this registry instead of importing individual model
classes. That keeps experiment configs swappable.
"""

from __future__ import annotations

import torch.nn as nn

from .clip_model import CLIPDetector
from .timm_model import TimmDetector


def build_model(config: dict) -> nn.Module:
    """Build a detector from a model config dictionary."""
    model_type = str(config.get("type", "timm")).lower()
    name = str(config.get("name", "resnet50"))
    common = {
        "name": name,
        "pretrained": bool(config.get("pretrained", True)),
        "num_classes": int(config.get("num_classes", 2)),
        "dropout": float(config.get("dropout", 0.0)),
        "freeze_backbone": bool(config.get("freeze_backbone", False)),
    }
    if model_type == "timm":
        return TimmDetector(**common)
    if model_type == "clip":
        common["freeze_backbone"] = bool(config.get("freeze_backbone", True))
        common["num_frozen_blocks"] = config.get("num_frozen_blocks")
        common["train_layer_norm"] = bool(config.get("train_layer_norm", True))
        return CLIPDetector(**common)
    raise ValueError(f"Unknown model type: {model_type}")
