"""Loss registry for config-driven training."""

from __future__ import annotations

import torch.nn as nn

from .bsce import BSCELoss
from .bsce_adaptive import AdaptiveBSCELoss
from .ce import BCEFromTwoLogitsLoss, CrossEntropyLoss
from .focal import FocalLoss
from .label_smoothing import LabelSmoothingCrossEntropyLoss


def build_loss(config: dict) -> nn.Module:
    """Build a loss from the `loss` section of a config."""
    name = str(config.get("name", "ce")).lower()
    if name in {"ce", "cross_entropy"}:
        return CrossEntropyLoss(reduction=config.get("reduction", "mean"))
    if name in {"bce", "bce_logits"}:
        return BCEFromTwoLogitsLoss(reduction=config.get("reduction", "mean"))
    if name in {"label_smoothing", "label_smoothing_ce"}:
        return LabelSmoothingCrossEntropyLoss(
            smoothing=float(config.get("smoothing", 0.1)),
            reduction=config.get("reduction", "mean"),
        )
    if name == "focal":
        return FocalLoss(gamma=float(config.get("gamma", 2.0)), reduction=config.get("reduction", "mean"))
    if name == "bsce":
        return BSCELoss(
            gamma=float(config.get("gamma", 1.0)),
            norm=float(config.get("norm", 2.0)),
            detach_weight=bool(config.get("detach_weight", False)),
            reduction=config.get("reduction", "mean"),
        )
    if name in {"bsce_adaptive", "adaptive_bsce"}:
        return AdaptiveBSCELoss(
            default_gamma=float(config.get("default_gamma", 1.0)),
            low_conf_gamma=float(config.get("low_conf_gamma", 0.5)),
            mid_conf_gamma=float(config.get("mid_conf_gamma", 0.2)),
            low_threshold=float(config.get("low_threshold", 0.2)),
            mid_threshold=float(config.get("mid_threshold", 0.5)),
            norm=float(config.get("norm", 2.0)),
            reduction=config.get("reduction", "mean"),
        )
    raise ValueError(f"Unknown loss: {name}")
