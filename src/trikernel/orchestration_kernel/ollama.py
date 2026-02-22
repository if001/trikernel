from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from trikernel.orchestration_kernel.protocols import LLMAPI

from .config import OllamaConfig, load_ollama_config
from .logging import get_logger
from .models import LLMResponse, LLMToolCall
from .message_builders import (
    build_messages,
    parse_response,
    parse_stream_message,
    to_langchain_tools,
)
from ..state_kernel.models import Task
from ..tool_kernel.structured_tool import TrikernelStructuredTool

logger = get_logger(__name__)


class OllamaLLM(LLMAPI):
    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.config = config or load_ollama_config()
        self.model = model or self.config.model
        self.timeout = timeout
        self._logger = get_logger("trikernel.ollama")
        self._last_response: Optional[LLMResponse] = None
        self._client = ChatOllama(
            model=self.model,
            base_url=self.config.base_url,
        )

    def generate(self, task: Task, tools: List[TrikernelStructuredTool]) -> LLMResponse:
        response = self._chat(task, tools, stream=False)
        self._last_response = response
        return response

    def stream_chunks(
        self, task: Task, tools: List[TrikernelStructuredTool]
    ) -> Iterable[str]:
        return self._chat_stream(task, tools)

    def collect_stream(
        self, task: Task, tools: List[TrikernelStructuredTool]
    ) -> Tuple[LLMResponse, List[str]]:
        chunks: List[str] = []
        for chunk in self.stream_chunks(task, tools):
            chunks.append(chunk)
        response = self._last_response or LLMResponse(user_output="", tool_calls=[])
        return response, chunks

    def _chat(
        self, task: Task, tools: List[TrikernelStructuredTool], stream: bool
    ) -> LLMResponse:
        self._logger.info("Ollama request model=%s stream=%s", self.model, stream)
        messages = build_messages(task)
        langchain_tools = to_langchain_tools(tools)
        if langchain_tools:
            llm_with_tools = self._client.bind_tools(langchain_tools)
            response = llm_with_tools.invoke(messages)
        else:
            response = self._client.invoke(messages)
        return parse_response(response)

    def _chat_stream(
        self, task: Task, tools: List[TrikernelStructuredTool]
    ) -> Iterable[str]:
        self._logger.info("Ollama stream model=%s", self.model)
        tool_calls: List[LLMToolCall] = []
        content_chunks: List[str] = []
        messages = build_messages(task)
        langchain_tools = to_langchain_tools(tools)
        if langchain_tools:
            llm_with_tools = self._client.bind_tools(langchain_tools)
            stream = llm_with_tools.stream(messages)
            llm_with_tools = self._client.bind_tools([])
        else:
            stream = self._client.stream(messages)
        for chunk in stream:
            chunk_content, calls = parse_stream_message(chunk)
            if chunk_content:
                content_chunks.append(chunk_content)
                yield chunk_content
            if calls:
                tool_calls.extend(calls)
        self._last_response = LLMResponse(
            user_output="".join(content_chunks), tool_calls=tool_calls
        )
