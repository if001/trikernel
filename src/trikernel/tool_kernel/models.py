from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .protocols import ToolLLMAPI


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
    llm_api: Optional["ToolLLMAPI"] = None


@dataclass
class ToolCall:
    tool_name: str
    args: Dict[str, Any]
