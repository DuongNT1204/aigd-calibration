"""PyTorch Lightning DataModule for AIGD experiments.

The DataModule centralizes all split paths and DataLoader settings, so trainers
do not need to know where JSONL files live.
"""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .dataset import AIGDImageDataset


class AIGDDataModule(pl.LightningDataModule):
    """LightningDataModule for train/val/test/predict splits."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        data_cfg = config["data"]
        self.train_path = data_cfg.get("train")
        self.val_path = data_cfg.get("val")
        self.test_id_path = data_cfg.get("test_id")
        self.test_ood_path = data_cfg.get("test_ood")
        self.predict_path = data_cfg.get("predict")
        self.image_size = int(data_cfg.get("image_size", 224))
        self.batch_size = int(data_cfg.get("batch_size", 32))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.normalization = str(data_cfg.get("normalization", "imagenet"))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.persistent_workers = bool(data_cfg.get("persistent_workers", self.num_workers > 0)) and self.num_workers > 0
        self.shuffle_train = bool(data_cfg.get("shuffle_train", True))
        self.drop_last_train = bool(data_cfg.get("drop_last_train", False))
        self.drop_last_eval = bool(data_cfg.get("drop_last_eval", False))

        self.train_dataset = None
        self.val_dataset = None
        self.test_id_dataset = None
        self.test_ood_dataset = None
        self.predict_dataset = None

    def _make_dataset(self, path: str | None, is_training: bool) -> AIGDImageDataset | None:
        if not path:
            return None
        return AIGDImageDataset(
            jsonl_path=Path(path),
            image_size=self.image_size,
            is_training=is_training,
            normalization=self.normalization,
        )

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for the requested Lightning stage."""
        if stage in (None, "fit"):
            self.train_dataset = self._make_dataset(self.train_path, is_training=True)
            self.val_dataset = self._make_dataset(self.val_path, is_training=False)
        if stage in (None, "test"):
            self.test_id_dataset = self._make_dataset(self.test_id_path, is_training=False)
            self.test_ood_dataset = self._make_dataset(self.test_ood_path, is_training=False)
        if stage in (None, "predict"):
            self.predict_dataset = self._make_dataset(self.predict_path or self.val_path, is_training=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_train,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last_train,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last_eval,
        )

    def test_dataloader(self) -> list[DataLoader]:
        loaders = []
        if self.test_id_dataset is not None:
            loaders.append(
                DataLoader(
                    self.test_id_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers,
                    drop_last=self.drop_last_eval,
                )
            )
        if self.test_ood_dataset is not None:
            loaders.append(
                DataLoader(
                    self.test_ood_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers,
                    drop_last=self.drop_last_eval,
                )
            )
        return loaders

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last_eval,
        )
