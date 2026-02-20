import asyncio
import time
from datetime import datetime, timedelta, timezone

from trikernel.execution.dispatcher import DispatchConfig, PendingWork, WorkDispatcher
from trikernel.execution.transports import ResultReceiver, WorkSender
from trikernel.state_kernel.kernel import StateKernel


class FakeSender(WorkSender):
    def __init__(self) -> None:
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class FakeReceiver(ResultReceiver):
    def __init__(self, payloads) -> None:
        self._payloads = list(payloads)

    async def recv_json(self):
        if not self._payloads:
            raise Exception("empty")
        return self._payloads.pop(0)

    async def try_recv_json(self):
        if not self._payloads:
            return None
        return self._payloads.pop(0)


def test_dispatch_respects_run_at(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    dispatcher = WorkDispatcher(
        state_api=state,
        work_sender=FakeSender(),
        result_receiver=FakeReceiver([]),
        config=DispatchConfig(),
    )
    task_id = state.task_create("work", {"message": "do"})
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    state.task_update(task_id, {"run_at": future.isoformat()})

    asyncio.run(dispatcher.run_once())
    assert dispatcher._pending == []


def test_send_pending_tracks_inflight(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    sender = FakeSender()
    dispatcher = WorkDispatcher(
        state_api=state,
        work_sender=sender,
        result_receiver=FakeReceiver([]),
        config=DispatchConfig(worker_count=1),
    )
    task_id = state.task_create("work", {"message": "do"})
    dispatcher._pending.append(
        PendingWork(task_id=task_id, enqueued_at=time.monotonic(), timeout_seconds=10)
    )
    asyncio.run(dispatcher._send_pending_tasks())
    assert sender.sent == [{"task_id": task_id}]
    assert task_id in dispatcher._inflight


def test_receive_results_finalize_task(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    receiver = FakeReceiver(
        [
            {
                "task_id": "t1",
                "task_state": "done",
                "user_output": "ok",
                "artifact_refs": [],
                "error": None,
            }
        ]
    )
    dispatcher = WorkDispatcher(
        state_api=state,
        work_sender=FakeSender(),
        result_receiver=receiver,
        config=DispatchConfig(),
    )
    task_id = state.task_create("work", {"message": "do"})
    dispatcher._inflight[task_id] = time.monotonic()
    asyncio.run(dispatcher._receive_worker_results())
    task = state.task_get(task_id)
    assert task is not None
    assert task.state == "done"
    notifications = state.task_list(task_type="notification", state=None)
    assert any(item.payload.get("message") == "ok" for item in notifications)


def test_fail_timed_out_pending(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    dispatcher = WorkDispatcher(
        state_api=state,
        work_sender=FakeSender(),
        result_receiver=FakeReceiver([]),
        config=DispatchConfig(work_queue_timeout_seconds=1),
    )
    task_id = state.task_create("work", {"message": "do"})
    dispatcher._pending.append(
        PendingWork(
            task_id=task_id,
            enqueued_at=time.monotonic() - 5,
            timeout_seconds=1,
        )
    )
    dispatcher._fail_timed_out_pending()
    task = state.task_get(task_id)
    assert task is not None
    assert task.state == "failed"
