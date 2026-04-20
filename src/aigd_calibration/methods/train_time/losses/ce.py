"""Cross-entropy and BCE-style losses.

The default detector returns two logits. BCE is implemented on the logit margin
fake_logit - real_logit, which is equivalent to binary logistic classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Standard two-class cross-entropy."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, labels.long())


class BCEFromTwoLogitsLoss(nn.Module):
    """Binary BCE using the margin between fake and real logits."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        margin = logits[:, 1] - logits[:, 0]
        return F.binary_cross_entropy_with_logits(margin, labels.float(), reduction=self.reduction)
