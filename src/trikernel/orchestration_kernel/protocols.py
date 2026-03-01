from __future__ import annotations

from typing import Generic, List, Protocol, Sequence, Tuple, TYPE_CHECKING, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from ..state_kernel.models import Task

if TYPE_CHECKING:
    from .models import LLMResponse, RunResult, RunnerContext


# class OrchestrationLLM(Protocol):
#     def generate(self, task: Task, tools: List[BaseTool]) -> "LLMResponse": ...
#
#     def collect_stream(
#         self, task: Task, tools: List[BaseTool]
#     ) -> Tuple["LLMResponse", List[str]]: ...
#
#     def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage: ...


class Runner(Protocol):
    def run(self, task: Task, runner_context: "RunnerContext") -> "RunResult": ...
