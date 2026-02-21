from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from trikernel.utils.logging import get_logger

from ..state_kernel.models import Task
from ..state_kernel.protocols import StateKernelAPI
from .transports import ResultReceiver, WorkSender, ZmqResultReceiver, ZmqWorkSender

logger = get_logger(__name__)


@dataclass
class DispatchConfig:
    worker_count: int = 2
    max_workers: int = 2
    poll_interval: float = 0.5
    zmq_endpoint: str = "inproc://trikernel-work"
    zmq_result_endpoint: str = "inproc://trikernel-work-results"
    worker_timeout_seconds: float = 60 * 10
    work_queue_timeout_seconds: float = 60 * 30


@dataclass
class PendingWork:
    task_id: str
    enqueued_at: float
    timeout_seconds: float


class WorkDispatcher:
    def __init__(
        self,
        state_api: StateKernelAPI,
        config: Optional[DispatchConfig] = None,
        work_sender: Optional[WorkSender] = None,
        result_receiver: Optional[ResultReceiver] = None,
    ) -> None:
        self.state_api = state_api
        self.config = config or DispatchConfig()
        self._work_sender = work_sender or ZmqWorkSender(self.config.zmq_endpoint)
        self._result_receiver = result_receiver or ZmqResultReceiver(
            self.config.zmq_result_endpoint
        )
        self._pending: List[PendingWork] = []
        self._inflight: Dict[str, float] = {}

    async def run_once(self) -> None:
        await self._dispatch_work_tasks()
        await self._send_pending_tasks()
        await self._receive_worker_results()
        self._fail_timed_out_pending()
        self._fail_timed_out_tasks()

    async def _dispatch_work_tasks(self) -> None:
        tasks = self.state_api.task_list(task_type="work", state="queued")
        now = datetime.now(timezone.utc)
        for task in tasks:
            if task.state != "queued":
                continue
            if self._is_already_tracked(task.task_id):
                continue
            run_at = _parse_run_at(task.payload or {})
            if run_at is None:
                continue
            if run_at > now:
                continue
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

    async def _send_pending_tasks(self) -> None:
        available = self.config.worker_count - len(self._inflight)
        if available <= 0:
            return
        pending = list(self._pending)
        self._pending = []
        for entry in pending:
            if available <= 0:
                self._pending.append(entry)
                continue
            await self._work_sender.send_json({"task_id": entry.task_id})
            self._inflight[entry.task_id] = time.monotonic()
            available -= 1

    async def _receive_worker_results(self) -> None:
        while True:
            try:
                payload = await self._result_receiver.try_recv_json()
            except Exception:
                logger.error("worker result receive failed", exc_info=True)
                break
            if payload is None:
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
                        "meta": payload.get("meta"),
                    },
                )
            self._finalize_task(task, payload)

    def _finalize_task(self, task: Task, payload: dict) -> None:
        task_state = payload.get("task_state") or "failed"
        if task_state == "done":
            if _is_recurring(task.payload or {}):
                self._reschedule_task(task)
            else:
                self.state_api.task_complete(task.task_id)
            return
        self.state_api.task_fail(
            task.task_id, payload.get("error") or {"message": "failed"}
        )

    def _reschedule_task(self, task: Task) -> None:
        self.state_api.task_update(task.task_id, _reschedule_patch(task.payload or {}))

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
            logger.error("worker timeout exceeded: %s", task_id)
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
                    logger.error("work queue timeout exceeded: %s", entry.task_id)
                continue
            still_pending.append(entry)
        self._pending = still_pending

    def _is_already_tracked(self, task_id: str) -> bool:
        if task_id in self._inflight:
            return True
        return any(entry.task_id == task_id for entry in self._pending)


def _parse_run_at(payload: dict) -> Optional[datetime]:
    run_at = payload.get("run_at")
    if not run_at:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(run_at))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_recurring(payload: dict) -> bool:
    return bool(payload.get("repeat_enabled") and payload.get("repeat_interval_seconds"))


def _clamp_repeat_interval(seconds: int) -> int:
    return max(3600, seconds)


def _next_run_at(interval_seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=interval_seconds)).isoformat()


def _reschedule_patch(payload: dict) -> dict:
    interval = _clamp_repeat_interval(int(payload.get("repeat_interval_seconds") or 0))
    next_payload = dict(payload)
    next_payload.update(
        {
            "run_at": _next_run_at(interval),
            "repeat_interval_seconds": interval,
            "repeat_enabled": True,
        }
    )
    return {
        "state": "queued",
        "claimed_by": None,
        "claim_expires_at": None,
        "payload": next_payload,
    }
