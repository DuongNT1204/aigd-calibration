"""Lightweight config validation.

This file intentionally avoids a heavy schema library at the base-code stage.
It checks the fields that would otherwise fail late in a training run.
"""

from __future__ import annotations


def require_keys(config: dict, keys: list[str], context: str) -> None:
    """Raise a clear error if required top-level keys are missing."""
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing required keys in {context}: {missing}")


def validate_train_config(config: dict) -> None:
    """Validate the minimum fields needed to train a model."""
    require_keys(config, ["experiment", "data", "model", "training", "loss"], "train config")
    require_keys(config["experiment"], ["name"], "experiment")
    require_keys(config["data"], ["train", "val"], "data")


def validate_calibration_config(config: dict) -> None:
    """Validate the minimum fields needed for post-hoc calibration."""
    require_keys(config, ["calibration"], "calibration config")
    require_keys(config["calibration"], ["method", "val_logits", "apply_logits"], "calibration")
