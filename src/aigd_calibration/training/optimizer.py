"""Optimizer builders.

The default path supports a BitMind-style parameter split: backbone parameters
can use a smaller learning rate while classifier-head parameters use a larger
learning rate.
"""

from __future__ import annotations

import torch


def split_param_groups(
    model: torch.nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    weight_decay_head: float,
) -> list[dict]:
    """Split trainable parameters into backbone and head groups."""
    head_params = []
    backbone_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head.") or ".head." in name:
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay_head})
    return groups


def build_optimizer(model_or_parameters, config: dict) -> torch.optim.Optimizer:
    """Build an optimizer from config."""
    name = str(config.get("name", "adamw")).lower()
    lr = float(config.get("lr", 1e-4))
    lr_backbone = float(config.get("lr_backbone", config.get("backbone_lr", lr)))
    lr_head = float(config.get("lr_head", config.get("head_lr", lr)))
    weight_decay = float(config.get("weight_decay", 1e-4))
    weight_decay_head = float(config.get("weight_decay_head", weight_decay))
    fused = bool(config.get("fused", False)) and torch.cuda.is_available()

    if hasattr(model_or_parameters, "named_parameters"):
        parameters = split_param_groups(
            model_or_parameters,
            lr_backbone=lr_backbone,
            lr_head=lr_head,
            weight_decay=weight_decay,
            weight_decay_head=weight_decay_head,
        )
    else:
        parameters = model_or_parameters

    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, fused=fused)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=float(config.get("momentum", 0.9)), weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")
