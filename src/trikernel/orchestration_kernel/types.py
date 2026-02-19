from __future__ import annotations

from typing import Optional, TypedDict


class ToolError(TypedDict):
    error_type: str
    message: str


class ToolResult(TypedDict, total=False):
    tool: str
    result: object
    tool_call_id: Optional[str]
