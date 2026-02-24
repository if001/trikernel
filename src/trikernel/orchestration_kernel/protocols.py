from __future__ import annotations

from typing import List, Protocol, Tuple, TYPE_CHECKING

from langchain_core.tools import BaseTool

from ..state_kernel.models import Task

if TYPE_CHECKING:
    from .models import LLMResponse, RunResult, RunnerContext


class OrchestrationLLM(Protocol):
    def generate(self, task: Task, tools: List[BaseTool]) -> "LLMResponse": ...

    def collect_stream(
        self, task: Task, tools: List[BaseTool]
    ) -> Tuple["LLMResponse", List[str]]: ...


class Runner(Protocol):
    def run(self, task: Task, runner_context: "RunnerContext") -> "RunResult": ...
