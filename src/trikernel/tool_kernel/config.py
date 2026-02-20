from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    small_model: str


def load_ollama_config() -> OllamaConfig:
    load_dotenv()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    small_model = os.environ.get("OLLAMA_SMALL_MODEL", "")
    return OllamaConfig(base_url=base_url, small_model=small_model)
