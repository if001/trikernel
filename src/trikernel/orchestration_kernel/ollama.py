from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from .config import OllamaConfig, load_ollama_config
from .logging import get_logger
from .models import LLMResponse, LLMToolCall
from ..state_kernel.models import Task


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

    def generate(self, task: Task, tools: List[Dict[str, Any]]) -> LLMResponse:
        response = self._chat(task, tools, stream=False)
        self._last_response = response
        return response

    def stream_chunks(self, task: Task, tools: List[Dict[str, Any]]) -> Iterable[str]:
        return self._chat_stream(task, tools)

    def collect_stream(
        self, task: Task, tools: List[Dict[str, Any]]
    ) -> Tuple[LLMResponse, List[str]]:
        chunks: List[str] = []
        for chunk in self.stream_chunks(task, tools):
            chunks.append(chunk)
        response = self._last_response or LLMResponse(user_output="", tool_calls=[])
        return response, chunks

    def _chat(
        self, task: Task, tools: List[Dict[str, Any]], stream: bool
    ) -> LLMResponse:
        tool_payload = _format_tools(tools)
        self._logger.info("Ollama request model=%s stream=%s", self.model, stream)
        messages = _build_messages(task)
        if tool_payload:
            response = self._client.invoke(messages, tools=tool_payload)
        else:
            response = self._client.invoke(messages)
        return _parse_response(response)

    def _chat_stream(self, task: Task, tools: List[Dict[str, Any]]) -> Iterable[str]:
        tool_payload = _format_tools(tools)
        self._logger.info("Ollama stream model=%s", self.model)
        tool_calls: List[LLMToolCall] = []
        content_chunks: List[str] = []
        messages = _build_messages(task)
        if tool_payload:
            stream = self._client.stream(messages, tools=tool_payload)
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


def _build_messages(task: Task) -> List[HumanMessage | AIMessage]:
    payload = task.payload or {}
    history = payload.get("history") or []
    messages: List[HumanMessage | AIMessage] = []
    for turn in history:
        user_message = turn.get("user_message")
        assistant_message = turn.get("assistant_message")
        if user_message:
            messages.append(HumanMessage(content=user_message))
        if assistant_message:
            messages.append(AIMessage(content=assistant_message))
    message = (
        payload.get("message")
        or payload.get("prompt")
        or json.dumps(payload, ensure_ascii=True)
    )
    messages.append(HumanMessage(content=message))
    return messages


def _format_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    for tool in tools:
        formatted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["tool_name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return formatted


def _parse_response(message: AIMessage) -> LLMResponse:
    content = message.content
    tool_calls = _parse_tool_calls(message)
    return LLMResponse(user_output=content, tool_calls=tool_calls)


def _parse_stream_message(
    message: AIMessageChunk,
) -> Tuple[str, List[LLMToolCall]]:
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
            function = tool_call.get("function", {})
            name = function.get("name")
            arguments = function.get("arguments", {})
        else:
            name = getattr(tool_call, "name", None) or getattr(tool_call, "tool", None)
            arguments = (
                getattr(tool_call, "args", None)
                or getattr(tool_call, "arguments", None)
                or {}
            )
        if name:
            calls.append(LLMToolCall(tool_name=name, args=arguments))
    return calls
