from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage

from ..utils.env import load_env
from .logging import get_logger
from .protocols import ToolLLMBase


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    small_model: str


def load_ollama_config() -> OllamaConfig:
    load_env()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    small_model = os.environ.get("OLLAMA_SMALL_MODEL", "")
    return OllamaConfig(base_url=base_url, small_model=small_model)


class ToolOllamaLLM(ToolLLMBase):
    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.config = config or load_ollama_config()
        self.model = model or self.config.small_model or "llama3"
        self.timeout = timeout
        self._logger = get_logger("trikernel.tool_ollama")
        self._client = ChatOllama(
            model=self.model,
            base_url=self.config.base_url,
        )

    def generate(self, prompt: str, tools: List[Any] | None = None) -> str:
        self._logger.info("Tool Ollama request model=%s", self.model)
        messages = [HumanMessage(content=prompt)]
        response = self._client.invoke(messages, config={"timeout": self.timeout})
        return _parse_response(response)


def _parse_response(message: AIMessage) -> str:
    return message.content or ""
