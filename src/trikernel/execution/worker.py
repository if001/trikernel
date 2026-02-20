from __future__ import annotations

from typing import Optional

from ..orchestration_kernel.models import RunResult, RunnerContext
from ..orchestration_kernel.protocols import LLMAPI, Runner
from ..state_kernel.models import Task
from ..state_kernel.protocols import StateKernelAPI
from ..tool_kernel.protocols import ToolAPI, ToolLLMAPI
from .transports import ResultSender, WorkReceiver, ZmqResultSender, ZmqWorkReceiver


class WorkWorker:
    def __init__(
        self,
        state_api: StateKernelAPI,
        tool_api: ToolAPI,
        runner: Runner,
        llm_api: LLMAPI,
        tool_llm_api: Optional[ToolLLMAPI],
        work_receiver: Optional[WorkReceiver] = None,
        result_sender: Optional[ResultSender] = None,
        work_endpoint: str = "inproc://trikernel-work",
        result_endpoint: str = "inproc://trikernel-work-results",
    ) -> None:
        self.state_api = state_api
        self.tool_api = tool_api
        self.runner = runner
        self.llm_api = llm_api
        self.tool_llm_api = tool_llm_api
        self._work_receiver = work_receiver or ZmqWorkReceiver(work_endpoint)
        self._result_sender = result_sender or ZmqResultSender(result_endpoint)

    async def run_once(self) -> None:
        payload = await self._work_receiver.try_recv_json()
        if payload is None:
            return
        task_id = payload.get("task_id")
        task = self.state_api.task_get(task_id) if task_id else None
        if not task:
            return
        task_meta = (task.payload or {}).get("meta")
        result = self._run_task(task, runner_id="worker")
        await self._result_sender.send_json(
            {
                "task_id": task.task_id,
                "task_state": result.task_state,
                "user_output": result.user_output,
                "artifact_refs": result.artifact_refs,
                "error": result.error,
                "meta": task_meta,
            }
        )

    def _run_task(self, task: Task, runner_id: str) -> RunResult:
        context = RunnerContext(
            runner_id=runner_id,
            state_api=self.state_api,
            tool_api=self.tool_api,
            llm_api=self.llm_api,
            tool_llm_api=self.tool_llm_api,
        )
        return self.runner.run(task, context)
