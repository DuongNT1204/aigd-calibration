"""Scheduler builders."""

from __future__ import annotations


def build_scheduler(optimizer, config: dict, total_steps: int | None = None):
    """Return a scheduler config or None."""
    name = str(config.get("name", "none")).lower()
    if name in {"none", ""}:
        return None
    if name == "cosine":
        import torch

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("t_max", 10)),
            eta_min=float(config.get("eta_min", 1e-6)),
        )
        return {"scheduler": scheduler, "interval": "epoch"}
    if name in {"step", "step_decay"}:
        import torch

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("step_size", 50)),
            gamma=float(config.get("gamma", 0.1)),
        )
        return {"scheduler": scheduler, "interval": "epoch"}
    if name in {"multistep", "multi_step"}:
        import torch

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(milestone) for milestone in config.get("milestones", [100, 150, 200])],
            gamma=float(config.get("gamma", 0.1)),
        )
        return {"scheduler": scheduler, "interval": "epoch"}
    if name in {"warmup_cosine", "warmup_cosine_restarts"}:
        if total_steps is None:
            raise ValueError("warmup_cosine scheduler requires total_steps")
        import torch

        warmup_ratio = float(config.get("warmup_ratio", 0.03))
        eta_min = float(config.get("eta_min", 1e-6))
        start_factor = float(config.get("start_factor", 0.01))
        min_warmup_steps = int(config.get("min_warmup_steps", 100))
        warmup_steps = min(max(min_warmup_steps, int(warmup_ratio * total_steps)), max(1, total_steps - 1))
        cosine_steps = max(1, total_steps - warmup_steps)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_steps,
            T_mult=int(config.get("t_mult", 1)),
            eta_min=eta_min,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
        return {"scheduler": scheduler, "interval": "step"}
    raise ValueError(f"Unknown scheduler: {name}")
