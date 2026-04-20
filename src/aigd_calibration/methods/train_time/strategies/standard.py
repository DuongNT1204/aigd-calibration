"""Standard single-model LightningModule.

Use this strategy for normal baselines and train-time loss methods:
CE, BCE, label smoothing, focal, BSCE, and adaptive BSCE.
"""

from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import torch

from aigd_calibration.evaluation.metrics import compute_binary_metrics
from aigd_calibration.models.registry import build_model
from aigd_calibration.methods.train_time.losses.registry import build_loss
from aigd_calibration.training.optimizer import build_optimizer
from aigd_calibration.training.scheduler import build_scheduler


class StandardAIGDModule(pl.LightningModule):
    """Single-detector training module."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model = build_model(config["model"])
        self.criterion = build_loss(config.get("loss", {"name": "ce"}))
        self.ece_bins = int(config.get("metrics", {}).get("ece_bins", 15))
        self.threshold = float(config.get("metrics", {}).get("threshold", 0.5))
        self._val_probs: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    @staticmethod
    def _unpack_batch(batch):
        """Support both dict batches and legacy (image, label) batches."""
        if isinstance(batch, dict):
            return batch["image"], batch["label"]
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise TypeError(f"Unsupported batch format: {type(batch)}")

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, labels = self._unpack_batch(batch)
        logits = self(images)
        loss = self.criterion(logits, labels)
        bs = labels.shape[0]
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, labels = self._unpack_batch(batch)
        logits = self(images)
        loss = self.criterion(logits, labels)
        bs = labels.shape[0]
        probs_fake = torch.softmax(logits, dim=-1)[:, 1]
        self._val_probs.append(probs_fake.detach().cpu())
        self._val_labels.append(labels.detach().cpu())
        self.log("val/loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_probs:
            return
        probs = torch.cat(self._val_probs).numpy()
        labels = torch.cat(self._val_labels).numpy()
        metrics = compute_binary_metrics(labels.astype(np.int64), probs, n_bins=self.ece_bins, threshold=self.threshold)
        for key, value in metrics.items():
            if value is not None:
                self.log(f"val/{key}", float(value), prog_bar=key in {"auc", "ece"}, sync_dist=True)
        self._val_probs.clear()
        self._val_labels.clear()

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, labels = self._unpack_batch(batch)
        logits = self(images)
        loss = self.criterion(logits, labels)
        probs_fake = torch.softmax(logits, dim=-1)[:, 1]
        self._val_probs.append(probs_fake.detach().cpu())
        self._val_labels.append(labels.detach().cpu())
        self.log("test/loss", loss, prog_bar=True, sync_dist=True, batch_size=labels.shape[0])
        return loss

    def on_test_epoch_end(self) -> None:
        if not self._val_probs:
            return
        probs = torch.cat(self._val_probs).numpy()
        labels = torch.cat(self._val_labels).numpy()
        metrics = compute_binary_metrics(labels.astype(np.int64), probs, n_bins=self.ece_bins, threshold=self.threshold)
        for key, value in metrics.items():
            if value is not None:
                self.log(f"test/{key}", float(value), prog_bar=key in {"auc", "ece"}, sync_dist=True)
        self._val_probs.clear()
        self._val_labels.clear()

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        images, _ = self._unpack_batch(batch)
        logits = self(images)
        return {"batch": batch, "logits": logits.detach().cpu()}

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model, self.config.get("optimizer", {}))
        scheduler = build_scheduler(optimizer, self.config.get("scheduler", {}), total_steps=self.trainer.estimated_stepping_batches)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
