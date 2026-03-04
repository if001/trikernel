from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama


from ...utils.env import load_env
from ...utils.logging import get_logger


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    small_model: str


def load_ollama_config() -> OllamaConfig:
    load_env()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "")
    small_model = os.environ.get("OLLAMA_SMALL_MODEL", model)
    return OllamaConfig(base_url=base_url, model=model, small_model=small_model)


class StateOllamaLLM:
    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        model: Optional[str] = None,
        timeout: int = 180,
    ) -> None:
        self.config = config or load_ollama_config()
        self.model = model or self.config.small_model or "llama3"
        self.timeout = timeout
        self._logger = get_logger("trikernel.state_ollama")
        self._client = ChatOllama(
            model=self.model,
            base_url=self.config.base_url,
        )

    def generate(self, prompt: str) -> str:
        self._logger.info("State Ollama request model=%s", self.model)
        config = {"timeout": self.timeout}
        response = self._client.invoke([HumanMessage(content=prompt)], config=config)
        return _parse_response(response)


def _parse_response(message: AIMessage) -> str:
    return message.content or ""
