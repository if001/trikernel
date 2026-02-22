from __future__ import annotations

import json
from typing import List, Sequence, Tuple

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool as LangchainStructuredTool

from ..logging import get_logger
from ..models import LLMResponse, LLMToolCall
from ..payloads import extract_llm_input
from ...state_kernel.models import Task
from ...tool_kernel.structured_tool import TrikernelStructuredTool

logger = get_logger(__name__)


def build_messages(task: Task) -> Sequence[BaseMessage]:
    payload = task.payload or {}
    llm_input = extract_llm_input(payload)
    if "messages" in llm_input:
        return llm_input["messages"]
    message = (
        llm_input.get("message")
        or llm_input.get("prompt")
        or json.dumps(llm_input, ensure_ascii=True)
    )
    return [HumanMessage(content=message)]


def to_langchain_tools(
    tools: Sequence[TrikernelStructuredTool],
) -> List[LangchainStructuredTool]:
    return [tool.as_langchain() for tool in tools]


def parse_response(message: AIMessage) -> LLMResponse:
    content = message.content
    tool_calls = parse_tool_calls(message)
    final_text = ""
    if isinstance(content, str):
        final_text = content
    else:
        logger.error("parse_response content not str: {content}")
    return LLMResponse(user_output=final_text, tool_calls=tool_calls, message=message)


def parse_stream_message(message: AIMessageChunk) -> Tuple[str, List[LLMToolCall]]:
    chunk = message.content or ""
    tool_calls = parse_tool_calls(message)
    return chunk, tool_calls


def parse_tool_calls(message: AIMessage | AIMessageChunk) -> List[LLMToolCall]:
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
