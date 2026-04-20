"""Post-hoc calibration registry."""

from __future__ import annotations

from .identity import IdentityCalibrator
from .temperature_scaling import TemperatureScaling


def build_calibrator(config: dict):
    """Build a post-hoc calibrator from config."""
    method = str(config.get("method", "identity")).lower()
    if method in {"identity", "none"}:
        return IdentityCalibrator()
    if method in {"temperature", "temperature_scaling"}:
        return TemperatureScaling(
            init_temperature=float(config.get("init_temperature", 1.0)),
            max_iter=int(config.get("max_iter", 100)),
        )
    raise ValueError(f"Unknown post-hoc calibration method: {method}")
