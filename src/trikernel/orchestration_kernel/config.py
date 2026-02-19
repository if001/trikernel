from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    small_model: str


def _find_env_path(start: Optional[Path] = None) -> Optional[Path]:
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def load_env(path: Optional[Path] = None) -> None:
    env_path = path or _find_env_path()
    if not env_path or not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_ollama_config() -> OllamaConfig:
    load_env()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "")
    small_model = os.environ.get("OLLAMA_SMALL_MODEL", model)
    return OllamaConfig(base_url=base_url, model=model, small_model=small_model)
