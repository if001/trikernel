from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = RichHandler(rich_tracebacks=True)
    formatter = logging.Formatter("%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_log_level(level: int) -> None:
    logging.getLogger().setLevel(level)
