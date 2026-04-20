"""Small logging setup for CLI commands."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure Python logging with a compact CLI-friendly format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
