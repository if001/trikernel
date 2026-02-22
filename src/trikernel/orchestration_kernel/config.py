from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    small_model: str


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str


def load_ollama_config() -> OllamaConfig:
    load_dotenv()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "")
    small_model = os.environ.get("OLLAMA_SMALL_MODEL", model)
    return OllamaConfig(base_url=base_url, model=model, small_model=small_model)


def load_gemini_config() -> GeminiConfig:
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    model = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
    return GeminiConfig(api_key=api_key, model=model)
