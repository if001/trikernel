from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMToolCall:
    tool_name: str
    args: Dict[str, Any]
    tool_call_id: Optional[str] = None


@dataclass
class LLMResponse:
    user_output: Optional[str]
    tool_calls: List[LLMToolCall] = field(default_factory=list)
    message: Optional[Any] = None


@dataclass
class RunResult:
    user_output: Optional[str]
    task_state: str
    artifact_refs: List[str] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    stream_chunks: List[str] = field(default_factory=list)
