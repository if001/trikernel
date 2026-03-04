from typing import Protocol

from trikernel.orchestration_kernel.models import RunResult
from trikernel.state_kernel.models import Task


class RunnerAPI(Protocol):
    def run(
        self,
        task: Task,
        *,
        conversation_id: str,
        stream: bool = False,
    ) -> RunResult: ...
