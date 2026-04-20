"""Label smoothing loss.

This is a simple train-time calibration baseline: it prevents the model from
learning overly sharp one-hot targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Cross-entropy with configurable label smoothing."""

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=float(smoothing), reduction=reduction)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, labels.long())
