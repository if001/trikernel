from __future__ import annotations

from typing import Protocol, List, Tuple

from ..state_kernel.models import Task
from ..tool_kernel.structured_tool import TrikernelStructuredTool

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    ## type checkのみ行う
    from .models import LLMResponse, RunnerContext, RunResult


class LLMAPI(Protocol):
    def generate(self, task: Task, tools: list[TrikernelStructuredTool]) -> LLMResponse: ...
    def collect_stream(
        self, task: Task, tools: List[TrikernelStructuredTool]
    ) -> Tuple[LLMResponse, List[str]]: ...


class Runner(Protocol):
    def run(self, task: Task, runner_context: RunnerContext) -> RunResult: ...
