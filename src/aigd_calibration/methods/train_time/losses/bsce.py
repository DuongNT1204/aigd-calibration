"""Brier-score-weighted cross entropy.

This follows the idea already present in the local `bitmind-image` project:
use the distance between the predicted probability vector and the one-hot target
as a weight on cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BSCELoss(nn.Module):
    """Brier-style weighted cross-entropy."""

    def __init__(self, gamma: float = 1.0, norm: float = 2.0, detach_weight: bool = False, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.norm = float(norm)
        self.detach_weight = bool(detach_weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long().view(-1, 1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        log_pt = log_probs.gather(1, labels).squeeze(1)
        one_hot = torch.zeros_like(probs).scatter_(1, labels, 1.0)

        weights = torch.norm(one_hot - probs, p=self.norm, dim=1) ** self.gamma
        if self.detach_weight:
            weights = weights.detach()
        loss = -weights * log_pt

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
