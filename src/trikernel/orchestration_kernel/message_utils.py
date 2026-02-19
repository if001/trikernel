from __future__ import annotations

import json
from typing import Dict, List, Sequence, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ..state_kernel.models import Turn
from .models import LLMResponse
from .types import ToolResult


HistoryItem = Union[Turn, Dict[str, object], BaseMessage]


def history_to_messages(history: Sequence[HistoryItem]) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    for turn in history:
        if isinstance(turn, BaseMessage):
            messages.append(turn)
            continue
        if isinstance(turn, dict):
            user_message = turn.get("user_message")
            assistant_message = turn.get("assistant_message")
        else:
            user_message = getattr(turn, "user_message", None)
            assistant_message = getattr(turn, "assistant_message", None)
        if isinstance(user_message, str) and user_message:
            messages.append(HumanMessage(content=user_message))
        if isinstance(assistant_message, str) and assistant_message:
            messages.append(AIMessage(content=assistant_message))
    return messages


def ensure_ai_message(response: LLMResponse) -> AIMessage:
    if isinstance(response.message, AIMessage):
        return response.message
    return AIMessage(content=response.user_output or "")


def tool_message_from_result(tool_result: ToolResult) -> ToolMessage:
    tool_call_id = str(tool_result.get("tool_call_id") or tool_result.get("tool") or "")
    content = json.dumps(tool_result.get("result"), ensure_ascii=False)
    return ToolMessage(content=content, tool_call_id=tool_call_id)


def messages_to_history(messages: Sequence[BaseMessage]) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for msg in messages:
        history.append({"role": msg.type, "content": str(msg.content)})
    return history
