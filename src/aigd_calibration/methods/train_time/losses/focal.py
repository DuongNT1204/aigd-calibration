"""Focal-style reweighting loss.

Focal loss is not primarily a calibration method, but it is useful to compare
because it changes how hard/uncertain samples influence optimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multiclass focal loss for two-class detector logits."""

    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long()
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        log_pt = log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
        pt = probs.gather(1, labels.view(-1, 1)).squeeze(1)
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
