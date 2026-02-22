from __future__ import annotations

import json
from typing import Dict, List, Optional

from langchain_core.messages import BaseMessage, ToolMessage

from ..logging import get_logger


logger = get_logger(__name__)


def safe_json_load(text: Optional[str]) -> Dict[str, object]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def add_budget_exceeded_message(tool_messages: List[BaseMessage]) -> None:
    tool_messages.append(
        ToolMessage(
            content='{"error":"上限に達したためtool使用をストップしました。ここまでのtoolの結果を利用し、調査が足りていない旨を含めて回答してください。","code":"BUDGET_EXCEEDED"}',
            tool_call_id="system",
        )
    )
