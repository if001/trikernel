from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

from ..utils.logging import get_logger
from .config import OllamaConfig, load_ollama_config


class StateOllamaLLM:
    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        model: Optional[str] = None,
        timeout: int = 60,
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
        response = self._client.invoke([HumanMessage(content=prompt)])
        return _parse_response(response)


def _parse_response(message: AIMessage) -> str:
    return message.content or ""
