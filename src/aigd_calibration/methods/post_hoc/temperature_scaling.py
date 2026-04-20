"""Temperature scaling for two-class logits.

Temperature is fit on validation logits only. Test logits are never used for
fitting, which keeps the experiment protocol clean.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from aigd_calibration.artifacts.logits import logits_to_prob_fake


class TemperatureScaling:
    """Single-parameter temperature scaling calibrator."""

    def __init__(self, init_temperature: float = 1.0, max_iter: int = 100) -> None:
        self.temperature = float(init_temperature)
        self.max_iter = int(max_iter)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaling":
        logits_t = torch.as_tensor(logits, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.long)
        log_temperature = torch.nn.Parameter(torch.log(torch.tensor(self.temperature, dtype=torch.float32)))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=self.max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature).clamp(min=0.05, max=20.0)
            loss = F.cross_entropy(logits_t / temperature, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temperature).detach().clamp(min=0.05, max=20.0))
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return logits_to_prob_fake(logits, temperature=self.temperature)

    def state_dict(self) -> dict:
        return {"method": "temperature", "temperature": self.temperature}
