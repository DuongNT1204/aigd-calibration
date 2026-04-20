"""YAML configuration loading utilities.

The loader returns a plain dictionary on purpose. That keeps the base code easy
to inspect and lets the schema module gradually become stricter as experiments
stabilize.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .defaults import DEFAULT_CONFIG


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two config dictionaries."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and apply project defaults."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    if not isinstance(user_config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    config = deep_merge(DEFAULT_CONFIG, user_config)
    config["_config_path"] = str(path)
    return config
