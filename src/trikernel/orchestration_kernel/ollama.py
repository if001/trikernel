from __future__ import annotations

import json
from typing import Iterable, List, Optional, Sequence, Tuple

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool as LangchainStructuredTool

from .config import OllamaConfig, load_ollama_config
from .logging import get_logger
from .models import LLMResponse, LLMToolCall
from .payloads import extract_llm_input
from ..state_kernel.models import Task
from ..tool_kernel.structured_tool import TrikernelStructuredTool

logger = get_logger(__name__)


class OllamaLLM:
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

    def stream_chunks(self, task: Task, tools: List[TrikernelStructuredTool]) -> Iterable[str]:
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
        messages = _build_messages(task)
        langchain_tools = _to_langchain_tools(tools)
        if langchain_tools:
            llm_with_tools = self._client.bind_tools(langchain_tools)
            response = llm_with_tools.invoke(messages)
        else:
            response = self._client.invoke(messages)
        return _parse_response(response)

    def _chat_stream(self, task: Task, tools: List[TrikernelStructuredTool]) -> Iterable[str]:
        self._logger.info("Ollama stream model=%s", self.model)
        tool_calls: List[LLMToolCall] = []
        content_chunks: List[str] = []
        messages = _build_messages(task)
        langchain_tools = _to_langchain_tools(tools)
        if langchain_tools:
            llm_with_tools = self._client.bind_tools(langchain_tools)
            stream = llm_with_tools.stream(messages)
            llm_with_tools = self._client.bind_tools([])
        else:
            stream = self._client.stream(messages)
        for chunk in stream:
            chunk_content, calls = _parse_stream_message(chunk)
            if chunk_content:
                content_chunks.append(chunk_content)
                yield chunk_content
            if calls:
                tool_calls.extend(calls)
        self._last_response = LLMResponse(
            user_output="".join(content_chunks), tool_calls=tool_calls
        )


def _build_messages(task: Task) -> Sequence[BaseMessage]:
    payload = task.payload or {}
    llm_input = extract_llm_input(payload)
    if "messages" in llm_input:
        return llm_input["messages"]
    history = llm_input.get("history") or []
    messages: List[HumanMessage | AIMessage] = []
    for turn in history:
        if isinstance(turn, (HumanMessage, AIMessage)):
            messages.append(turn)
            continue
        user_message = turn.get("user_message")
        assistant_message = turn.get("assistant_message")
        if user_message:
            messages.append(HumanMessage(content=user_message))
        if assistant_message:
            messages.append(AIMessage(content=assistant_message))

    message = (
        llm_input.get("message")
        or llm_input.get("prompt")
        or json.dumps(llm_input, ensure_ascii=True)
    )
    messages.append(HumanMessage(content=message))
    return messages


def _to_langchain_tools(
    tools: Sequence[TrikernelStructuredTool],
) -> List[LangchainStructuredTool]:
    return [tool.as_langchain() for tool in tools]


def _parse_response(message: AIMessage) -> LLMResponse:
    content = message.content
    tool_calls = _parse_tool_calls(message)
    final_text = ""
    if isinstance(content, str):
        final_text = content
    else:
        logger.error("parse_response content not str: {content}")
    return LLMResponse(user_output=final_text, tool_calls=tool_calls, message=message)


def _parse_stream_message(message: AIMessageChunk) -> Tuple[str, List[LLMToolCall]]:
    chunk = message.content or ""
    tool_calls = _parse_tool_calls(message)
    return chunk, tool_calls


def _parse_tool_calls(message: AIMessage | AIMessageChunk) -> List[LLMToolCall]:
    calls = []
    raw_calls = getattr(message, "tool_calls", None) or message.additional_kwargs.get(
        "tool_calls", []
    )
    for tool_call in raw_calls or []:
        if isinstance(tool_call, dict):
            call_id = tool_call.get("id")
            if "function" in tool_call:
                function = tool_call.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments", {})
            else:
                name = tool_call.get("name", "")
                arguments = tool_call.get("args", {})
        else:
            name = getattr(tool_call, "name", None) or getattr(tool_call, "tool", None)
            arguments = (
                getattr(tool_call, "args", None)
                or getattr(tool_call, "arguments", None)
                or {}
            )
            call_id = getattr(tool_call, "id", None) or getattr(
                tool_call, "tool_call_id", None
            )
        if name:
            calls.append(
                LLMToolCall(tool_name=name, args=arguments, tool_call_id=call_id)
            )
    return calls
