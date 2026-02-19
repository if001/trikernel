from __future__ import annotations

from typing import Protocol

from ..state_kernel.models import Task

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    ## type checkのみ行う
    from .models import LLMResponse, RunnerContext, RunResult


class LLMAPI(Protocol):
    def generate(self, task: Task, tools: list[dict]) -> "LLMResponse": ...


class Runner(Protocol):
    def run(self, task: Task, runner_context: "RunnerContext") -> "RunResult": ...
