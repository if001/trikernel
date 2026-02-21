from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional

from ..state_kernel.protocols import StateKernelAPI
from ..tool_kernel.protocols import ToolAPI, ToolLLMAPI
from .protocols import LLMAPI


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
class RunnerContext:
    runner_id: str
    state_api: StateKernelAPI
    tool_api: ToolAPI
    llm_api: LLMAPI
    tool_llm_api: ToolLLMAPI | None = None
    stream: bool = False


@dataclass
class RunResult:
    user_output: Optional[str]
    task_state: str
    artifact_refs: List[str] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    stream_chunks: List[str] = field(default_factory=list)


@dataclass
class Budget:
    remaining_steps: int
    spent_steps: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "remaining_steps": self.remaining_steps,
            "spent_steps": self.spent_steps,
        }


@dataclass
class SimpleStepContext:
    tool_summary: str = ""
    budget: Budget = field(default_factory=lambda: Budget(remaining_steps=5))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_result": self.tool_summary,
            "budget": self.budget.to_dict(),
        }

    def to_str(self) -> str:
        return (
            f"- tool_summary: {self.tool_summary}\n"
            f"- remaining_step: {self.budget.remaining_steps}\n"
            f"- spent_steps: {self.budget.spent_steps}\n"
        )


@dataclass
class StepContext:
    facts: List[str] = field(default_factory=list)
    open_issues: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    last_result: str = ""
    budget: Budget = field(default_factory=lambda: Budget(remaining_steps=5))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": list(self.facts),
            "open_issues": list(self.open_issues),
            "plan": list(self.plan),
            "last_result": self.last_result,
            "budget": self.budget.to_dict(),
        }

    def to_str(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
