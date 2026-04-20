"""Common helpers used inside LightningModules."""

from __future__ import annotations

import torch


def probs_fake_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return fake-class probability from two-class logits."""
    return torch.softmax(logits, dim=-1)[:, 1]


def predictions_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return hard class predictions from two-class logits."""
    return torch.argmax(logits, dim=-1)
