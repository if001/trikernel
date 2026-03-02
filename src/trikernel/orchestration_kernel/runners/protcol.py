from typing import Protocol

from trikernel.orchestration_kernel.models import RunResult, RunnerContext
from trikernel.state_kernel.models import Task


class RunnerAPI(Protocol):
    def run(self, task: Task, runner_context: RunnerContext) -> RunResult: ...
