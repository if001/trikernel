from __future__ import annotations

from typing import Any, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage

from trikernel.tool_kernel.protocols import ToolLLMAPI


from .config import OllamaConfig, load_ollama_config
from .logging import get_logger


class ToolOllamaLLM(ToolLLMAPI):
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
        if tools:
            llm_with_tools = self._client.bind_tools(tools)
            response = llm_with_tools.invoke(messages)
        else:
            response = self._client.invoke(messages)
        return _parse_response(response)


def _parse_response(message: AIMessage) -> str:
    return message.content or ""
