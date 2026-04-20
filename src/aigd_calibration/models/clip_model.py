"""CLIP vision encoder detector.

The CLIP backbone is useful for cross-generator experiments because prior work
shows pretrained feature spaces often generalize better than narrow detectors.
"""

from __future__ import annotations

import torch

from .base import DetectorBase
from .heads import build_linear_head


class CLIPDetector(DetectorBase):
    """CLIP image encoder plus a linear classifier head."""

    def __init__(
        self,
        name: str = "openai/clip-vit-large-patch14",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.0,
        freeze_backbone: bool = True,
        num_frozen_blocks: int | None = None,
        train_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import CLIPVisionConfig, CLIPVisionModel
        except ImportError as exc:
            raise ImportError("Install transformers to use CLIPDetector.") from exc

        if pretrained:
            self.vision = CLIPVisionModel.from_pretrained(name)
        else:
            self.vision = CLIPVisionModel(CLIPVisionConfig.from_pretrained(name))

        hidden_size = int(self.vision.config.hidden_size)
        self.head = build_linear_head(hidden_size, num_classes=num_classes, dropout=dropout)

        if freeze_backbone:
            self.freeze_layers(num_frozen_blocks=num_frozen_blocks, train_layer_norm=train_layer_norm)

    def freeze_layers(self, num_frozen_blocks: int | None = None, train_layer_norm: bool = True) -> None:
        """Freeze CLIP vision layers.

        If num_frozen_blocks is None, the whole CLIP vision encoder is frozen.
        If it is an integer, the first N transformer blocks are frozen and the
        remaining blocks are trainable. This mirrors the BitMind training setup.
        """
        for parameter in self.vision.parameters():
            parameter.requires_grad = False

        if num_frozen_blocks is not None:
            layers = self.vision.vision_model.encoder.layers
            num_frozen_blocks = max(0, min(int(num_frozen_blocks), len(layers)))
            for layer in layers[num_frozen_blocks:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

            if train_layer_norm:
                for name in ("pre_layernorm", "layernorm", "final_layer_norm"):
                    module = getattr(self.vision.vision_model, name, None)
                    if module is not None and hasattr(module, "parameters"):
                        for parameter in module.parameters():
                            parameter.requires_grad = True

        for parameter in self.head.parameters():
            parameter.requires_grad = True

        self._apply_train_eval_modes()

    def _apply_train_eval_modes(self) -> None:
        """Put trainable vision submodules in train mode, frozen ones in eval mode."""
        # Start with entire vision in eval, then selectively enable train mode
        # for submodules that contain at least one trainable parameter.
        self.vision.eval()
        for module in self.vision.modules():
            if any(p.requires_grad for p in module.parameters()):
                module.train()
        self.head.train()

    def train(self, mode: bool = True) -> "CLIPDetector":
        """Put trainable parts in train mode; keep frozen parts in eval mode."""
        super().train(mode)
        if mode:
            self._apply_train_eval_modes()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.vision(pixel_values=images)
        cls_token = output.pooler_output
        return self.head(cls_token)
