from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from trikernel.utils.logging import get_logger

from ..orchestration_kernel.models import RunResult, RunnerContext
from ..orchestration_kernel.protocols import LLMAPI, Runner
from ..tool_kernel.protocols import ToolLLMAPI
from ..state_kernel.models import Task
from ..state_kernel.protocols import StateKernelAPI
from ..tool_kernel.kernel import ToolAPI
from .transports import (
    ResultReceiver,
    ResultSender,
    WorkReceiver,
    WorkSender,
    ZmqResultReceiver,
    ZmqResultSender,
    ZmqWorkReceiver,
    ZmqWorkSender,
)

logger = get_logger(__name__)


@dataclass
class CompositionConfig:
    runner_id: str = "main"
    worker_count: int = 2
    max_workers: int = 2
    poll_interval: float = 0.5
    zmq_endpoint: str = "inproc://trikernel-work"
    zmq_result_endpoint: str = "inproc://trikernel-work-results"
    worker_timeout_seconds: float = 60 * 10  # 10min
    work_queue_timeout_seconds: float = 60 * 30  # 30min


@dataclass
class PendingWork:
    task_id: str
    enqueued_at: float
    timeout_seconds: float


class CompositionRuntime:
    def __init__(
        self,
        state_api: StateKernelAPI,
        tool_api: ToolAPI,
        runner: Runner,
        llm_api: LLMAPI,
        tool_llm_api: Optional[ToolLLMAPI],
        config: Optional[CompositionConfig] = None,
        work_sender: Optional[WorkSender] = None,
        work_receiver: Optional[WorkReceiver] = None,
        result_sender: Optional[ResultSender] = None,
        result_receiver: Optional[ResultReceiver] = None,
    ) -> None:
        self.state_api = state_api
        self.tool_api = tool_api
        self.runner = runner
        self.llm_api = llm_api
        self.tool_llm_api = tool_llm_api
        self.config = config or CompositionConfig()
        self._work_sender = work_sender or ZmqWorkSender(self.config.zmq_endpoint)
        self._work_receiver = work_receiver or ZmqWorkReceiver(self.config.zmq_endpoint)
        self._result_sender = result_sender or ZmqResultSender(
            self.config.zmq_result_endpoint
        )
        self._result_receiver = result_receiver or ZmqResultReceiver(
            self.config.zmq_result_endpoint
        )
        self._stop = asyncio.Event()
        self._worker_tasks: List[asyncio.Task] = []
        self._main_task: Optional[asyncio.Task] = None
        self._pending: List[PendingWork] = []
        self._inflight: Dict[str, float] = {}

    async def start(self) -> None:
        self._ensure_worker_count()
        logger.info("main start")
        self._main_task = asyncio.create_task(self._main_loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._main_task:
            await self._main_task
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

    def _ensure_worker_count(self) -> None:
        worker_total = min(self.config.worker_count, self.config.max_workers)
        for _ in range(worker_total):
            self._worker_tasks.append(asyncio.create_task(self._worker_loop()))

    async def _main_loop(self) -> None:
        logger.info("_main_loop")
        while not self._stop.is_set():
            logger.info("_dispatch_work_tasks()")
            await self._dispatch_work_tasks()
            logger.info("self._send_pending_tasks(sender)")
            await self._send_pending_tasks(self._work_sender)
            logger.info("self._receive_worker_results(result_receiver)")
            await self._receive_worker_results(self._result_receiver)
            self._fail_timed_out_pending()
            self._fail_timed_out_tasks()
            await asyncio.sleep(self.config.poll_interval)

    async def _dispatch_work_tasks(self) -> None:
        tasks = self.state_api.task_list(task_type="work", state="queued")
        now = datetime.now(timezone.utc)
        for task in tasks:
            if task.state != "queued":
                continue
            if self._is_already_tracked(task.task_id):
                continue
            run_at = self._parse_run_at(task)
            if run_at is None:
                continue
            if run_at > now:
                continue
            logger.info("claim worker task")
            claimed = self.state_api.task_claim({"task_id": task.task_id}, "main", 30)
            if not claimed:
                continue
            self._pending.append(
                PendingWork(
                    task_id=task.task_id,
                    enqueued_at=time.monotonic(),
                    timeout_seconds=self.config.work_queue_timeout_seconds,
                )
            )

    async def _send_pending_tasks(self, sender: Any) -> None:
        available = self.config.worker_count - len(self._inflight)
        if available <= 0:
            return
        pending = list(self._pending)
        self._pending = []
        for entry in pending:
            if available <= 0:
                self._pending.append(entry)
                continue
            await sender.send_json({"task_id": entry.task_id})
            self._inflight[entry.task_id] = time.monotonic()
            available -= 1

    async def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                payload = await self._work_receiver.recv_json()
            except asyncio.CancelledError:
                logger.error("worker_loop asyncio.CancelledError")
                return
            task_id = payload.get("task_id")
            task = self.state_api.task_get(task_id) if task_id else None
            if not task:
                logger.error("task not found")
                continue
            logger.info("run worker")
            result = self._run_task(task, runner_id="worker")
            await self._result_sender.send_json(
                {
                    "task_id": task.task_id,
                    "task_state": result.task_state,
                    "user_output": result.user_output,
                    "artifact_refs": result.artifact_refs,
                    "error": result.error,
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

    def _finalize_task(self, task: Task, result: RunResult) -> None:
        if result.task_state == "done":
            self.state_api.task_complete(task.task_id)
        else:
            self.state_api.task_fail(
                task.task_id, result.error or {"message": "failed"}
            )

    async def _receive_worker_results(self, receiver: ResultReceiver) -> None:
        while True:
            try:
                payload = await receiver.recv_json()
            except asyncio.CancelledError:
                logger.error("_recive_worker_results asyncio.CancelledError")
                return
            except Exception as e:
                logger.error(f"_recive_worker_results {e}")
                break
            task_id = payload.get("task_id")
            if not task_id:
                continue
            self._inflight.pop(task_id, None)
            task = self.state_api.task_get(task_id)
            if not task:
                continue
            user_output = payload.get("user_output")
            if user_output:
                self.state_api.task_create(
                    "notification",
                    {
                        "message": user_output,
                        "severity": "info",
                        "related_task_id": task.task_id,
                        "artifact_refs": payload.get("artifact_refs") or [],
                    },
                )
            else:
                logger.warning("Worker task ended. But output is None")
            result = RunResult(
                user_output=payload.get("user_output"),
                task_state=payload.get("task_state") or "failed",
                artifact_refs=payload.get("artifact_refs") or [],
                error=payload.get("error"),
                stream_chunks=[],
            )
            self._finalize_task(task, result)

    async def run_once(self) -> None:
        await self._dispatch_work_tasks()
        await self._send_pending_tasks(self._work_sender)
        await self._receive_worker_results(self._result_receiver)
        self._fail_timed_out_pending()
        self._fail_timed_out_tasks()

    def _fail_timed_out_tasks(self) -> None:
        if self.config.worker_timeout_seconds <= 0:
            return
        now = time.monotonic()
        timed_out = [
            task_id
            for task_id, started_at in self._inflight.items()
            if now - started_at > self.config.worker_timeout_seconds
        ]
        for task_id in timed_out:
            self._inflight.pop(task_id, None)
            task = self.state_api.task_get(task_id)
            if not task:
                continue
            self.state_api.task_fail(
                task.task_id,
                {"code": "WORKER_TIMEOUT", "message": "Worker timeout exceeded."},
            )
            self._pending = [
                entry for entry in self._pending if entry.task_id != task_id
            ]

    def _fail_timed_out_pending(self) -> None:
        if self.config.work_queue_timeout_seconds <= 0:
            return
        now = time.monotonic()
        still_pending: List[PendingWork] = []
        for entry in self._pending:
            if now - entry.enqueued_at > entry.timeout_seconds:
                task = self.state_api.task_get(entry.task_id)
                if task:
                    self.state_api.task_fail(
                        task.task_id,
                        {
                            "code": "WORK_QUEUE_TIMEOUT",
                            "message": "Work queue timeout exceeded.",
                        },
                    )
                continue
            still_pending.append(entry)
        self._pending = still_pending

    def _parse_run_at(self, task: Task) -> Optional[datetime]:
        run_at = task.run_at
        if not run_at:
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            parsed = datetime.fromisoformat(str(run_at))
        except (TypeError, ValueError):
            logger.error("_parse_run_at: parse error")
            self.state_api.task_fail(
                task.task_id,
                {"code": "INVALID_RUN_AT", "message": "Invalid run_at timestamp."},
            )
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _is_already_tracked(self, task_id: str) -> bool:
        if task_id in self._inflight:
            return True
        return any(entry.task_id == task_id for entry in self._pending)
