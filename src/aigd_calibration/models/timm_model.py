"""timm-based CNN/ViT detector models.

This covers ResNet, ConvNeXt, ViT, EVA, and other timm architectures through
one wrapper that replaces the classifier with a two-class head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import DetectorBase
from .heads import build_linear_head


class TimmDetector(DetectorBase):
    """A two-class detector built from any timm backbone."""

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm to use TimmDetector.") from exc

        # num_classes=0 makes timm return features instead of classifier logits.
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        in_features = int(self.backbone.num_features)
        self.head = build_linear_head(in_features, num_classes=num_classes, dropout=dropout)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.head(features)
