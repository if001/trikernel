from __future__ import annotations

from typing import Optional, Sequence

from trikernel.utils.logging import get_logger

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.store.base import BaseStore

from ..orchestration_kernel.models import RunResult, RunnerContext
from ..state_kernel.memory_manager import LangMemMemoryManager
from ..orchestration_kernel.protocols import OrchestrationLLM, Runner
from ..state_kernel.models import Task
from ..state_kernel.protocols import StateKernelAPI, MessageStoreAPI
from ..tool_kernel.kernel import ToolKernel
from ..tool_kernel.protocols import ToolLLMBase
from .transports import ResultSender, WorkReceiver, ZmqResultSender, ZmqWorkReceiver

logger = get_logger(__name__)


class WorkWorker:
    def __init__(
        self,
        state_api: StateKernelAPI,
        message_store: MessageStoreAPI,
        tool_api: ToolKernel,
        runner: Runner,
        llm_api: OrchestrationLLM,
        tool_llm_api: ToolLLMBase,
        memory_manager: LangMemMemoryManager,
        store: BaseStore,
        work_receiver: Optional[WorkReceiver] = None,
        result_sender: Optional[ResultSender] = None,
        work_endpoint: str = "inproc://trikernel-work",
        result_endpoint: str = "inproc://trikernel-work-results",
    ) -> None:
        self.state_api = state_api
        self.message_store = message_store
        self.tool_api = tool_api
        self.runner = runner
        self.llm_api = llm_api
        self.tool_llm_api = tool_llm_api
        self._memory_manager = memory_manager
        self._store = store
        self._work_receiver = work_receiver or ZmqWorkReceiver(work_endpoint)
        self._result_sender = result_sender or ZmqResultSender(result_endpoint)
        if hasattr(self.state_api, "set_memory_store"):
            self.state_api.set_memory_store(store)

    async def run_once(self) -> None:
        payload = await self._work_receiver.try_recv_json()
        if payload is None:
            return
        task_id = payload.get("task_id")
        task = self.state_api.task_get(task_id) if task_id else None
        if not task:
            return
        task_meta = (task.payload or {}).get("meta")
        if _is_memory_task(task):
            result = await self._run_memory_task(task)
        else:
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
            conversation_id="default",
            state_api=self.state_api,
            message_store=self.message_store,
            tool_api=self.tool_api,
            llm_api=self.llm_api,
            tool_llm_api=self.tool_llm_api,
            store=self._store,
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

    async def _run_memory_task(self, task: Task) -> RunResult:
        payload = task.payload or {}
        conversation_id = str(payload.get("conversation_id") or "default")
        messages = _coerce_messages(payload.get("messages"))
        if not messages:
            return RunResult(
                user_output=None,
                task_state="failed",
                error={
                    "code": "MEMORY_TASK_INVALID",
                    "message": "Memory task requires messages.",
                },
            )
        try:
            await self._memory_manager.update(
                messages,
                conversation_id=conversation_id,
            )
        except Exception:
            logger.error("memory task failed: %s", task.task_id, exc_info=True)
            return RunResult(
                user_output=None,
                task_state="failed",
                error={"code": "MEMORY_TASK_FAILED", "message": "Memory task failed."},
            )
        return RunResult(user_output=None, task_state="done")


def _is_memory_task(task: Task) -> bool:
    payload = task.payload or {}
    return payload.get("kind") == "memory_update"


def _coerce_messages(raw_messages: object) -> Sequence[BaseMessage]:
    if not isinstance(raw_messages, list):
        return []
    messages: list[BaseMessage] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, str):
            continue
        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages
