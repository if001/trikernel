import asyncio
from datetime import datetime, timezone

from trikernel.execution.dispatcher import DispatchConfig, WorkDispatcher
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


def test_recurring_reschedules_on_done(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    task_id = state.task_create("work", {"message": "do"})
    state.task_update(
        task_id,
        {
            "payload": {
                "repeat_enabled": True,
                "repeat_interval_seconds": 10,
            },
        },
    )
    receiver = FakeReceiver(
        [
            {
                "task_id": task_id,
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
    dispatcher._inflight[task_id] = 0.0
    asyncio.run(dispatcher._receive_worker_results())

    task = state.task_get(task_id)
    assert task is not None
    assert task.state == "queued"
    run_at = task.payload.get("run_at")
    assert run_at is not None
    run_at_time = datetime.fromisoformat(run_at)
    assert run_at_time.tzinfo is not None
    assert run_at_time > datetime.now(timezone.utc)
