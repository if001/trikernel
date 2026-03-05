from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langmem.short_term import RunningSummary
from typing_extensions import Annotated, TypedDict


@dataclass
class Budget:
    remaining_steps: int
    spent_steps: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "remaining_steps": self.remaining_steps,
            "spent_steps": self.spent_steps,
        }


@dataclass
class SimpleStepContext:
    role: str = "main"
    task_type: str = ""
    budget: Budget = field(default_factory=lambda: Budget(remaining_steps=5))

    def to_str(self) -> str:
        return (
            f"- role: {self.role}\n"
            f"- task_type: {self.task_type}\n"
            f"- remaining_step: {self.budget.remaining_steps}\n"
            f"- spent_steps: {self.budget.spent_steps}\n"
        )


@dataclass
class ToolStepContext:
    last_observation: str = ""
    error_summary: str = ""
    need_clarification: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    budget: Budget = field(default_factory=lambda: Budget(remaining_steps=5))

    def to_str(self) -> str:
        return (
            f"- ツール利用の結果: {self.notes}\n"
            f"- last_observation: {self.last_observation}\n"
            f"- error_summary: {self.error_summary}\n"
            f"- need_clarification: {self.need_clarification}\n"
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": list(self.facts),
            "open_issues": list(self.open_issues),
            "plan": list(self.plan),
            "last_result": self.last_result,
            "budget": self.budget.to_dict(),
        }

    def to_str(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class BaseState(TypedDict):
    task_id: str
    runtime_id: str
    messages: Annotated[list[BaseMessage], add_messages]


class ToolLoopState(BaseState):
    tool_set: set[str]
    stop: bool
    memory_context_text: str


class SimpleToolLoopState(ToolLoopState):
    step_context: SimpleStepContext
    running_summary: RunningSummary | None
    tool_set: set[str]
    stop: bool
    memory_context_text: str
    tool_results: list[str]


class DeepToolLoopState(BaseState):
    tool_set: set[str]
    stop: bool
    memory_context_text: str
    tool_step_context: ToolStepContext
    phase: str
    phase_goal: str
    running_summary: RunningSummary | None
