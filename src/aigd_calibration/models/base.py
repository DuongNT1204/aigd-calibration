"""Shared model interfaces.

All detector models should return two-class logits shaped [batch, 2]. Keeping
that contract stable makes losses, evaluation, calibration, and Diff-DML simple.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DetectorBase(nn.Module):
    """Base class documenting the detector forward contract."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return two-class logits for real/fake classification."""
        raise NotImplementedError
