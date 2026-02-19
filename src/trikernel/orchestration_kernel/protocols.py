from __future__ import annotations

from typing import Any, Protocol, List, Tuple
from langchain_core.tools import StructuredTool

from ..state_kernel.models import Task

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    ## type checkのみ行う
    from .models import LLMResponse, RunnerContext, RunResult


class LLMAPI(Protocol):
    def generate(self, task: Task, tools: list[StructuredTool]) -> LLMResponse: ...
    def collect_stream(
        self, task: Task, tools: List[Any]
    ) -> Tuple[LLMResponse, List[str]]: ...


class Runner(Protocol):
    def run(self, task: Task, runner_context: RunnerContext) -> RunResult: ...
