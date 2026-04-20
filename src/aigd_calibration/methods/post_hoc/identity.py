"""Identity calibrator.

This is useful as a baseline post-hoc method: it leaves probabilities unchanged
but still follows the calibrator interface.
"""

from __future__ import annotations

import numpy as np

from aigd_calibration.artifacts.logits import logits_to_prob_fake


class IdentityCalibrator:
    """No-op calibrator."""

    temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "IdentityCalibrator":
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return logits_to_prob_fake(logits, temperature=1.0)

    def state_dict(self) -> dict:
        return {"method": "identity", "temperature": 1.0}
