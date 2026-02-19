from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ToolDefinition:
    tool_name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    effects: List[str] = field(default_factory=list)


@dataclass
class ToolContext:
    runner_id: str
    task_id: str | None
    state_api: Any
    now: str


@dataclass
class ToolCall:
    tool_name: str
    args: Dict[str, Any]
