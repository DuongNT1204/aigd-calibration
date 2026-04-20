"""Adaptive BSCE loss.

The adaptive version lets the exponent depend on how confident the model is in
the correct class. The thresholds are intentionally configurable for research.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveBSCELoss(nn.Module):
    """BSCE with per-sample gamma chosen from confidence thresholds."""

    def __init__(
        self,
        default_gamma: float = 1.0,
        low_conf_gamma: float = 0.5,
        mid_conf_gamma: float = 0.2,
        low_threshold: float = 0.2,
        mid_threshold: float = 0.5,
        norm: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.default_gamma = float(default_gamma)
        self.low_conf_gamma = float(low_conf_gamma)
        self.mid_conf_gamma = float(mid_conf_gamma)
        self.low_threshold = float(low_threshold)
        self.mid_threshold = float(mid_threshold)
        self.norm = float(norm)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long().view(-1, 1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        log_pt = log_probs.gather(1, labels).squeeze(1)
        pt = probs.gather(1, labels).squeeze(1)
        one_hot = torch.zeros_like(probs).scatter_(1, labels, 1.0)

        with torch.no_grad():
            gamma = torch.full_like(pt, self.default_gamma)
            gamma = torch.where(pt < self.mid_threshold, torch.full_like(pt, self.mid_conf_gamma), gamma)
            gamma = torch.where(pt < self.low_threshold, torch.full_like(pt, self.low_conf_gamma), gamma)
            weights = torch.norm(one_hot - probs, p=self.norm, dim=1) ** gamma

        loss = -weights * log_pt
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
