from __future__ import annotations

import json
from typing import Dict, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .models import LLMResponse
from .types import ToolResult


def ensure_ai_message(response: LLMResponse) -> AIMessage:
    if isinstance(response.message, AIMessage):
        return response.message
    return AIMessage(content=response.user_output or "")


def tool_message_from_result(tool_result: ToolResult) -> ToolMessage:
    tool_call_id = str(tool_result.get("tool_call_id") or tool_result.get("tool") or "")
    content = json.dumps(tool_result.get("result"), ensure_ascii=False)
    return ToolMessage(content=content, tool_call_id=tool_call_id)

