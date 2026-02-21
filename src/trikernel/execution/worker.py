from __future__ import annotations

from typing import Optional

from trikernel.utils.logging import get_logger

from ..orchestration_kernel.models import RunResult, RunnerContext
from ..orchestration_kernel.protocols import LLMAPI, Runner
from ..state_kernel.models import Task
from ..state_kernel.protocols import StateKernelAPI
from ..tool_kernel.protocols import ToolAPI, ToolLLMAPI
from .transports import ResultSender, WorkReceiver, ZmqResultSender, ZmqWorkReceiver

logger = get_logger(__name__)


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
        try:
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
        except Exception:
            logger.error("worker result send failed: %s", task.task_id, exc_info=True)
            self.state_api.task_fail(
                task.task_id,
                {"code": "WORKER_SEND_FAILED", "message": "Failed to send result."},
            )

    def _run_task(self, task: Task, runner_id: str) -> RunResult:
        context = RunnerContext(
            runner_id=runner_id,
            state_api=self.state_api,
            tool_api=self.tool_api,
            llm_api=self.llm_api,
            tool_llm_api=self.tool_llm_api,
        )
        try:
            return self.runner.run(task, context)
        except Exception:
            logger.error("worker task failed: %s", task.task_id, exc_info=True)
            return RunResult(
                user_output=None,
                task_state="failed",
                error={"code": "WORKER_EXCEPTION", "message": "Worker failed."},
            )
