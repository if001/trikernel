from __future__ import annotations

from dataclasses import dataclass
import asyncio
import concurrent.futures
from datetime import datetime, timedelta, timezone
import threading
from typing import Any, Dict, List, Optional, Union

from trikernel.utils.logging import get_logger

from .dispatcher import DispatchConfig, WorkDispatcher
from .worker import WorkWorker
from .loop import ExecutionLoop, LoopConfig
from ..orchestration_kernel.models import RunResult, RunnerContext
from ..orchestration_kernel.protocols import LLMAPI, Runner
from ..state_kernel.models import Task
from ..state_kernel.protocols import StateKernelAPI
from ..tool_kernel.protocols import ToolAPI, ToolLLMAPI
from .payloads import UserRequestPayload, WorkPayload

logger = get_logger(__name__)


@dataclass
class MessageResult:
    message: Optional[str]
    task_state: str
    artifact_refs: List[str]
    error: Optional[Dict[str, object]]
    stream_chunks: List[str]


class TrikernelSession:
    def __init__(
        self,
        state_api: StateKernelAPI,
        tool_api: ToolAPI,
        runner: Runner,
        llm_api: LLMAPI,
        tool_llm_api: Optional[ToolLLMAPI] = None,
        conversation_id: str = "default",
        runner_id: str = "main",
        claim_ttl_seconds: int = 30,
        main_runner_timeout_seconds: int = 60 * 10,
    ) -> None:
        self._state_api = state_api
        self._tool_api = tool_api
        self._runner = runner
        self._llm_api = llm_api
        self._tool_llm_api = tool_llm_api
        self._conversation_id = conversation_id
        self._runner_id = runner_id
        self._claim_ttl_seconds = claim_ttl_seconds
        self._main_runner_timeout_seconds = main_runner_timeout_seconds
        self._runtime_loop: Optional[asyncio.AbstractEventLoop] = None
        self._runtime_thread: Optional[threading.Thread] = None
        self._dispatcher: Optional[WorkDispatcher] = None
        self._worker: Optional[WorkWorker] = None
        self._loop: Optional[ExecutionLoop] = None
        self._loop_task: Optional[asyncio.Task] = None

    def send_message(self, message: str, stream: bool = False) -> MessageResult:
        task_id = self._state_api.task_create(
            "user_request", UserRequestPayload(user_message=message).to_dict()
        )
        turn_id = self._state_api.turn_append_user(
            self._conversation_id, message, task_id
        )
        claimed_id = self._state_api.task_claim(
            {"task_id": task_id}, self._runner_id, self._claim_ttl_seconds
        )
        if not claimed_id:
            logger.error("failed to claim task: %s", task_id)
            self._state_api.task_fail(
                task_id,
                {"code": "CLAIM_FAILED", "message": "Failed to claim task."},
            )
            return MessageResult(
                message=None,
                task_state="failed",
                artifact_refs=[],
                error={"message": "Failed to claim task."},
                stream_chunks=[],
            )
        task = self._state_api.task_get(claimed_id)
        if not task:
            logger.error("failed to load task: %s", claimed_id)
            self._state_api.task_fail(
                claimed_id,
                {"code": "TASK_NOT_FOUND", "message": "Failed to load task."},
            )
            return MessageResult(
                message=None,
                task_state="failed",
                artifact_refs=[],
                error={"message": "Failed to load task."},
                stream_chunks=[],
            )
        result = self._run_task_with_timeout(task, stream=stream)
        assistant_message = result.user_output or ""
        if result.stream_chunks:
            assistant_message = "".join(result.stream_chunks) or assistant_message
        self._finalize_task(task, result)
        self._state_api.turn_set_assistant(
            turn_id,
            assistant_message,
            result.artifact_refs,
            {"task_state": result.task_state},
        )
        return MessageResult(
            message=assistant_message,
            task_state=result.task_state,
            artifact_refs=result.artifact_refs,
            error=result.error,
            stream_chunks=result.stream_chunks,
        )

    def drain_notifications(self) -> List[str]:
        messages: List[str] = []
        while True:
            notification_id = self._state_api.task_claim(
                {"task_type": "notification"}, self._runner_id, self._claim_ttl_seconds
            )
            if not notification_id:
                break
            notification = self._state_api.task_get(notification_id)
            if not notification:
                continue
            payload = notification.payload or {}
            message = payload.get("message")
            if isinstance(message, str) and message:
                messages.append(message)
            self._state_api.task_complete(notification_id)
        return messages

    def create_work_task(
        self,
        payload: Union[WorkPayload, Dict[str, Any]],
        run_at: Optional[str] = None,
        repeat_every_seconds: Optional[int] = None,
        repeat_enabled: bool = False,
    ) -> str:
        payload_dict = (
            payload.to_dict() if isinstance(payload, WorkPayload) else payload
        )
        if run_at:
            _validate_run_at(run_at)
            payload_dict["run_at"] = run_at
        if repeat_every_seconds is not None:
            repeat_seconds = max(3600, int(repeat_every_seconds))
            payload_dict["repeat_interval_seconds"] = repeat_seconds
            payload_dict["repeat_enabled"] = bool(repeat_enabled)
        elif repeat_enabled:
            payload_dict["repeat_enabled"] = True
        return self._state_api.task_create("work", payload_dict)

    def start_workers(
        self,
        dispatch_config: Optional[DispatchConfig] = None,
        loop_config: Optional[LoopConfig] = None,
    ) -> None:
        if self._runtime_thread and self._runtime_thread.is_alive():
            logger.info("already start...")
            return
        logger.info("start worker")
        self._dispatcher = WorkDispatcher(self._state_api, config=dispatch_config)
        self._worker = WorkWorker(
            state_api=self._state_api,
            tool_api=self._tool_api,
            runner=self._runner,
            llm_api=self._llm_api,
            tool_llm_api=self._tool_llm_api,
        )
        self._loop = ExecutionLoop(self._dispatcher, self._worker, loop_config)

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._runtime_loop = loop
            self._loop_task = loop.create_task(self._loop.run_forever())
            loop.run_forever()

        self._runtime_thread = threading.Thread(target=_run, daemon=True)
        self._runtime_thread.start()

    def stop_workers(self) -> None:
        if not self._runtime_loop:
            return
        for task in (self._loop_task,):
            if task:
                task.cancel()
        self._runtime_loop.call_soon_threadsafe(self._runtime_loop.stop)
        if self._runtime_thread:
            self._runtime_thread.join(timeout=5)
        self._runtime_loop = None
        self._runtime_thread = None
        if self._loop:
            self._loop.stop()
        self._dispatcher = None
        self._worker = None
        self._loop = None
        self._loop_task = None

    def _run_task(self, task: Task, stream: bool) -> RunResult:
        context = RunnerContext(
            runner_id=self._runner_id,
            state_api=self._state_api,
            tool_api=self._tool_api,
            llm_api=self._llm_api,
            tool_llm_api=self._tool_llm_api,
            stream=stream,
        )
        try:
            return self._runner.run(task, context)
        except Exception:
            logger.error("main task failed: %s", task.task_id, exc_info=True)
            return RunResult(
                user_output=None,
                task_state="failed",
                error={"code": "RUNNER_EXCEPTION", "message": "Runner failed."},
            )

    def _run_task_with_timeout(self, task: Task, stream: bool) -> RunResult:
        timeout_seconds = self._main_runner_timeout_seconds
        if timeout_seconds <= 0:
            return self._run_task(task, stream=stream)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_task, task, stream)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                logger.error("main runner timeout: %s", task.task_id)
                return RunResult(
                    user_output=None,
                    task_state="failed",
                    error={"code": "MAIN_TIMEOUT", "message": "Runner timeout."},
                )

    def _finalize_task(self, task: Task, result: RunResult) -> None:
        if result.task_state == "done":
            self._state_api.task_complete(task.task_id)
        else:
            self._state_api.task_fail(
                task.task_id, result.error or {"message": "failed"}
            )


def _validate_run_at(run_at: str) -> None:
    try:
        parsed = datetime.fromisoformat(run_at)
    except ValueError as exc:
        raise ValueError("run_at must be ISO8601 format") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if parsed < now:
        raise ValueError("run_at must be in the future")
    if parsed > now + timedelta(days=365):
        raise ValueError("run_at must be within 1 year")
