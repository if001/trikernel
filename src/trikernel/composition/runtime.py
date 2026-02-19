from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional

from ..orchestration_kernel.models import RunResult, RunnerContext
from ..state_kernel.models import Task
from ..state_kernel.protocols import StateKernelAPI
from ..tool_kernel.kernel import ToolKernel


@dataclass
class CompositionConfig:
    runner_id: str = "main"
    worker_count: int = 2
    max_workers: int = 2
    poll_interval: float = 0.5
    zmq_endpoint: str = "inproc://trikernel-work"


class CompositionRuntime:
    def __init__(
        self,
        state_api: StateKernelAPI,
        tool_api: ToolKernel,
        runner: Any,
        llm_api: Any,
        config: Optional[CompositionConfig] = None,
    ) -> None:
        self.state_api = state_api
        self.tool_api = tool_api
        self.runner = runner
        self.llm_api = llm_api
        self.config = config or CompositionConfig()
        self._stop = asyncio.Event()
        self._worker_tasks: List[asyncio.Task] = []
        self._main_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._ensure_worker_count()
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
        sender = self._create_sender()
        while not self._stop.is_set():
            await self._dispatch_work_tasks(sender)
            await self._process_user_tasks()
            await asyncio.sleep(self.config.poll_interval)

    async def _dispatch_work_tasks(self, sender: Any) -> None:
        task_id = self.state_api.task_claim({"task_type": "work"}, "main", 30)
        if not task_id:
            return
        await sender.send_json({"task_id": task_id})

    async def _process_user_tasks(self) -> None:
        for task_type in ("user_request", "notification"):
            task_id = self.state_api.task_claim({"task_type": task_type}, "main", 30)
            if not task_id:
                continue
            task = self.state_api.task_get(task_id)
            if not task:
                continue
            if task.task_type == "notification":
                self.state_api.task_complete(task.task_id)
                continue
            result = self._run_task(task, runner_id="main")
            self._finalize_task(task, result)

    async def _worker_loop(self) -> None:
        receiver = self._create_receiver()
        while not self._stop.is_set():
            try:
                payload = await receiver.recv_json(flags=0)
            except asyncio.CancelledError:
                return
            task_id = payload.get("task_id")
            task = self.state_api.task_get(task_id) if task_id else None
            if not task:
                continue
            print("run worker task", task)
            result = self._run_task(task, runner_id="worker")
            if result.user_output:
                self.state_api.task_create(
                    "notification",
                    {
                        "message": result.user_output,
                        "severity": "info",
                        "related_task_id": task.task_id,
                        "artifact_refs": result.artifact_refs,
                    },
                )
            self._finalize_task(task, result)

    def _run_task(self, task: Task, runner_id: str) -> RunResult:
        context = RunnerContext(
            runner_id=runner_id,
            state_api=self.state_api,
            tool_api=self.tool_api,
            llm_api=self.llm_api,
        )
        return self.runner.run(task, context)

    def _finalize_task(self, task: Task, result: RunResult) -> None:
        if result.task_state == "done":
            self.state_api.task_complete(task.task_id)
        else:
            self.state_api.task_fail(
                task.task_id, result.error or {"message": "failed"}
            )

    def _create_sender(self) -> Any:
        try:
            import zmq  # type: ignore
            import zmq.asyncio  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyzmq is required for the composition layer") from exc
        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.PUSH)
        socket.bind(self.config.zmq_endpoint)
        return socket

    def _create_receiver(self) -> Any:
        try:
            import zmq  # type: ignore
            import zmq.asyncio  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyzmq is required for the composition layer") from exc
        context = zmq.asyncio.Context.instance()
        socket = context.socket(zmq.PULL)
        socket.connect(self.config.zmq_endpoint)
        return socket
