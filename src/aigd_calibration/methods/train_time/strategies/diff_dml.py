"""Differentiated Deep Mutual Learning LightningModule.

Diff-DML is a training strategy, not a normal loss. It trains a primary detector
and an auxiliary detector together with KL agreement while reporting/evaluating
only the primary detector.
"""

from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F

from aigd_calibration.evaluation.metrics import compute_binary_metrics
from aigd_calibration.methods.train_time.losses.registry import build_loss
from aigd_calibration.models.registry import build_model
from aigd_calibration.training.optimizer import build_optimizer
from aigd_calibration.training.scheduler import build_scheduler


class DiffDMLModule(pl.LightningModule):
    """Two-model Diff-DML strategy."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        model_cfg = config["model"]
        if "primary" in model_cfg:
            primary_cfg = model_cfg["primary"]
            auxiliary_cfg = model_cfg.get("auxiliary", primary_cfg)
        else:
            primary_cfg = model_cfg
            auxiliary_cfg = config.get("model_aux", primary_cfg)
        self.model_f = build_model(primary_cfg)
        self.model_g = build_model(auxiliary_cfg)
        self.criterion = build_loss(config.get("loss", {"name": "ce"}))
        self.kl_weight = float(config.get("training", {}).get("kl_weight", 1.0))
        self.ece_bins = int(config.get("metrics", {}).get("ece_bins", 15))
        self.threshold = float(config.get("metrics", {}).get("threshold", 0.5))
        self.automatic_optimization = False
        scheduler_cfg_f = config.get("scheduler", {})
        scheduler_cfg_g = config.get("scheduler_aux", scheduler_cfg_f)
        self.scheduler_interval_f = self._scheduler_interval(scheduler_cfg_f)
        self.scheduler_interval_g = self._scheduler_interval(scheduler_cfg_g)
        self._val_probs: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward only through the primary model for eval/export."""
        return self.model_f(images)

    @staticmethod
    def _unpack_batch(batch):
        """Support both dict batches and legacy (image, label) batches."""
        if isinstance(batch, dict):
            return batch["image"], batch["label"]
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise TypeError(f"Unsupported batch format: {type(batch)}")

    @staticmethod
    def _kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """KL teacher -> student with detached teacher outside caller."""
        return F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )

    @staticmethod
    def _scheduler_interval(config: dict) -> str | None:
        """Return how a scheduler should be stepped in manual optimization."""
        name = str(config.get("name", "none")).lower()
        if name in {"none", ""}:
            return None
        if name in {"warmup_cosine", "warmup_cosine_restarts"}:
            return "step"
        return "epoch"

    def _configured_schedulers(self) -> tuple[object | None, object | None]:
        """Return schedulers aligned as (model_f, model_g)."""
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return None, None
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]

        scheduler_f = None
        scheduler_g = None
        index = 0
        if self.scheduler_interval_f is not None:
            scheduler_f = schedulers[index]
            index += 1
        if self.scheduler_interval_g is not None:
            scheduler_g = schedulers[index]
        return scheduler_f, scheduler_g

    def _step_schedulers(self, interval: str) -> None:
        """Step schedulers manually because Diff-DML uses manual optimization."""
        scheduler_f, scheduler_g = self._configured_schedulers()
        if self.scheduler_interval_f == interval and scheduler_f is not None:
            scheduler_f.step()
        if self.scheduler_interval_g == interval and scheduler_g is not None:
            scheduler_g.step()

    def training_step(self, batch: dict, batch_idx: int) -> None:
        opt_f, opt_g = self.optimizers()
        images, labels = self._unpack_batch(batch)

        logits_f = self.model_f(images)
        logits_g = self.model_g(images)

        loss_ce_f = self.criterion(logits_f, labels)
        loss_kl_f = self._kl(logits_f, logits_g.detach())
        loss_f = loss_ce_f + self.kl_weight * loss_kl_f
        loss_g = self._kl(logits_g, logits_f.detach())

        opt_f.zero_grad()
        self.manual_backward(loss_f)
        opt_f.step()

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()

        self._step_schedulers("step")

        bs = labels.shape[0]
        self.log("train/loss_f", loss_f, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("train/loss_ce_f", loss_ce_f, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("train/loss_kl_f", loss_kl_f, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("train/loss_g", loss_g, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)

    def on_train_epoch_end(self) -> None:
        self._step_schedulers("epoch")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, labels = self._unpack_batch(batch)
        logits = self.model_f(images)
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
        logits = self.model_f(images)
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
        logits = self.model_f(images)
        return {"batch": batch, "logits": logits.detach().cpu()}

    def configure_optimizers(self):
        opt_f = build_optimizer(self.model_f, self.config.get("optimizer", {}))
        opt_g = build_optimizer(self.model_g, self.config.get("optimizer_aux", self.config.get("optimizer", {})))
        scheduler_f = build_scheduler(opt_f, self.config.get("scheduler", {}), total_steps=self.trainer.estimated_stepping_batches)
        scheduler_g = build_scheduler(
            opt_g,
            self.config.get("scheduler_aux", self.config.get("scheduler", {})),
            total_steps=self.trainer.estimated_stepping_batches,
        )
        schedulers = [scheduler for scheduler in (scheduler_f, scheduler_g) if scheduler is not None]
        if not schedulers:
            return [opt_f, opt_g]
        return [opt_f, opt_g], schedulers
