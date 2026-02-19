from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..state_kernel.protocols import StateKernelAPI
from ..tool_kernel.protocols import ToolAPI
from .protocols import LLMAPI


@dataclass
class LLMToolCall:
    tool_name: str
    args: Dict[str, Any]


@dataclass
class LLMResponse:
    user_output: Optional[str]
    tool_calls: List[LLMToolCall] = field(default_factory=list)


@dataclass
class RunnerContext:
    runner_id: str
    state_api: StateKernelAPI
    tool_api: ToolAPI
    llm_api: LLMAPI
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
class StepContext:
    facts: List[str] = field(default_factory=list)
    open_issues: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    last_result: str = ""
    artifact_refs: List[str] = field(default_factory=list)
    budget: Budget = field(default_factory=lambda: Budget(remaining_steps=3))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": list(self.facts),
            "open_issues": list(self.open_issues),
            "plan": list(self.plan),
            "last_result": self.last_result,
            "artifact_refs": list(self.artifact_refs),
            "budget": self.budget.to_dict(),
        }
