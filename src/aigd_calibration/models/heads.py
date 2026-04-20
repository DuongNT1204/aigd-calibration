"""Reusable classifier heads."""

from __future__ import annotations

import torch.nn as nn


def build_linear_head(in_features: int, num_classes: int = 2, dropout: float = 0.0) -> nn.Module:
    """Build a small dropout + linear classifier head."""
    layers: list[nn.Module] = []
    if dropout > 0:
        layers.append(nn.Dropout(p=float(dropout)))
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)
