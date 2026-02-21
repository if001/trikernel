from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(
    log_path: Optional[Path] = None,
    console_level: int = logging.INFO,
) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_path = log_path or _default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.ERROR)
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root.addHandler(file_handler)

    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("%(name)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)


def set_log_level(level: int) -> None:
    if not _CONFIGURED:
        setup_logging()
    for handler in logging.getLogger().handlers:
        if isinstance(handler, RichHandler):
            handler.setLevel(level)


def _default_log_path() -> Path:
    env_path = os.environ.get("TRIKERNEL_LOG_PATH")
    if env_path:
        return Path(env_path)
    env_dir = os.environ.get("TRIKERNEL_LOG_DIR")
    if env_dir:
        return Path(env_dir) / "trikernel.log"
    return Path("logs") / "trikernel.log"
