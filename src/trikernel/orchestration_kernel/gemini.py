from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI

from trikernel.orchestration_kernel.protocols import LLMAPI

from .config import GeminiConfig, load_gemini_config
from .logging import get_logger
from .message_builders import (
    build_messages,
    parse_response,
    parse_stream_message,
    to_langchain_tools,
)
from .models import LLMResponse, LLMToolCall
from ..state_kernel.models import Task
from ..tool_kernel.structured_tool import TrikernelStructuredTool

logger = get_logger(__name__)


class GeminiLLM(LLMAPI):
    def __init__(
        self,
        config: Optional[GeminiConfig] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.config = config or load_gemini_config()
        self.model = model or self.config.model
        self.timeout = timeout
        self._logger = get_logger("trikernel.gemini")
        self._last_response: Optional[LLMResponse] = None
        self._client = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.config.api_key,
        )

    def generate(self, task: Task, tools: List[TrikernelStructuredTool]) -> LLMResponse:
        response = self._chat(task, tools)
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

    def _chat(self, task: Task, tools: List[TrikernelStructuredTool]) -> LLMResponse:
        self._logger.info("Gemini request model=%s", self.model)
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
        self._logger.info("Gemini stream model=%s", self.model)
        tool_calls: List[LLMToolCall] = []
        content_chunks: List[str] = []
        messages = build_messages(task)
        langchain_tools = to_langchain_tools(tools)
        if langchain_tools:
            llm_with_tools = self._client.bind_tools(langchain_tools)
            stream = llm_with_tools.stream(messages)
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
