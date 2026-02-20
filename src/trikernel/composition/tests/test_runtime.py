import asyncio
import time
from datetime import datetime, timedelta, timezone

from trikernel.composition.runtime import (
    CompositionConfig,
    CompositionRuntime,
    PendingWork,
)
from trikernel.state_kernel.kernel import StateKernel
from trikernel.composition.transports import ResultReceiver, WorkSender
from trikernel.tool_kernel.protocols import ToolAPI


class DummyRunner:
    def run(self, task, context):
        raise AssertionError("run should not be called in this test")


class DummyToolAPI(ToolAPI):
    def tool_register(self, tool_definition, handler) -> None:
        return None

    def tool_register_structured(self, tool) -> None:
        return None

    def tool_describe(self, tool_name):
        raise KeyError(tool_name)

    def tool_search(self, query):
        return []

    def tool_invoke(self, tool_name, args, tool_context):
        raise KeyError(tool_name)

    def tool_list(self):
        return []

    def tool_descriptions(self):
        return []

    def tool_structured_list(self):
        return []


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


def test_dispatch_work_respects_run_at(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    runtime = CompositionRuntime(
        state_api=state,
        tool_api=DummyToolAPI(),
        runner=DummyRunner(),
        llm_api=None,
        tool_llm_api=None,
        config=CompositionConfig(),
        result_receiver=FakeReceiver([]),
    )
    task_id = state.task_create("work", {"message": "do"})
    run_at = datetime.now(timezone.utc) + timedelta(hours=1)
    state.task_update(task_id, {"run_at": run_at.isoformat()})

    asyncio.run(runtime._dispatch_work_tasks())
    assert runtime._pending == []

    state.task_update(
        task_id, {"run_at": datetime.min.replace(tzinfo=timezone.utc).isoformat()}
    )
    asyncio.run(runtime._dispatch_work_tasks())
    assert len(runtime._pending) == 1


def test_send_pending_tasks_tracks_inflight(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    runtime = CompositionRuntime(
        state_api=state,
        tool_api=DummyToolAPI(),
        runner=DummyRunner(),
        llm_api=None,
        tool_llm_api=None,
        config=CompositionConfig(worker_count=1),
    )
    task_id = state.task_create("work", {"message": "do"})
    runtime._pending.append(
        PendingWork(task_id=task_id, enqueued_at=time.monotonic(), timeout_seconds=10)
    )
    sender = FakeSender()
    asyncio.run(runtime._send_pending_tasks(sender))
    assert sender.sent == [{"task_id": task_id}]
    assert task_id in runtime._inflight


def test_receive_worker_results_finalize_task(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    runtime = CompositionRuntime(
        state_api=state,
        tool_api=DummyToolAPI(),
        runner=DummyRunner(),
        llm_api=None,
        tool_llm_api=None,
        config=CompositionConfig(),
        result_receiver=FakeReceiver([]),
    )
    task_id = state.task_create("work", {"message": "do"})
    runtime._inflight[task_id] = time.monotonic()
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
    asyncio.run(runtime._receive_worker_results(receiver))
    task = state.task_get(task_id)
    assert task is not None
    assert task.state == "done"
    notifications = state.task_list(task_type="notification", state=None)
    assert any(item.payload.get("message") == "ok" for item in notifications)


def test_fail_timed_out_pending(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    runtime = CompositionRuntime(
        state_api=state,
        tool_api=DummyToolAPI(),
        runner=DummyRunner(),
        llm_api=None,
        tool_llm_api=None,
        config=CompositionConfig(work_queue_timeout_seconds=1),
        result_receiver=FakeReceiver([]),
    )
    task_id = state.task_create("work", {"message": "do"})
    runtime._pending.append(
        PendingWork(
            task_id=task_id,
            enqueued_at=time.monotonic() - 5,
            timeout_seconds=1,
        )
    )
    runtime._fail_timed_out_pending()
    task = state.task_get(task_id)
    assert task is not None
    assert task.state == "failed"
