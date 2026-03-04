import asyncio

from trikernel.execution.dispatcher import DispatchConfig, WorkDispatcher
from trikernel.execution.worker import WorkWorker
from trikernel.execution.transports import (
    ResultReceiver,
    ResultSender,
    WorkReceiver,
    WorkSender,
)
from trikernel.state_kernel import LangMemMemoryManager, StateKernel, build_memory_store
from trikernel.orchestration_kernel.models import RunResult


class InMemoryChannel(WorkSender, WorkReceiver, ResultSender, ResultReceiver):
    def __init__(self):
        self._queue = []

    async def send_json(self, payload):
        self._queue.append(payload)

    async def recv_json(self):
        if not self._queue:
            raise Exception("empty")
        return self._queue.pop(0)

    async def try_recv_json(self):
        if not self._queue:
            return None
        return self._queue.pop(0)


class DummyRunner:
    def run(self, task, *, conversation_id: str, stream: bool = False):
        return RunResult(user_output="done", task_state="done")


class FailingRunner:
    def run(self, task, *, conversation_id: str, stream: bool = False):
        raise RuntimeError("boom")


def test_work_task_end_to_end(tmp_path):
    async def _run():
        state = StateKernel(data_dir=tmp_path)
        async with build_memory_store(data_dir=tmp_path) as store:
            work_channel = InMemoryChannel()
            result_channel = InMemoryChannel()
            dispatcher = WorkDispatcher(
                state_api=state,
                work_sender=work_channel,
                result_receiver=result_channel,
                config=DispatchConfig(worker_count=1, poll_interval=0),
            )
            worker = WorkWorker(
                state_api=state,
                runner=DummyRunner(),
                memory_manager=LangMemMemoryManager(store),
                work_receiver=work_channel,
                result_sender=result_channel,
            )

            task_id = state.task_create(
                "work", {"message": "do", "meta": {"channel_id": 1}}
            )

            await dispatcher.run_once()
            await worker.run_once()
            await dispatcher.run_once()

            task = state.task_get(task_id)
            assert task is not None
            assert task.state == "done"
            notifications = state.task_list(task_type="notification", state=None)
            assert notifications
            assert notifications[0].payload.get("meta", {}).get("channel_id") == 1

    asyncio.run(_run())


def test_work_task_failure_marks_failed(tmp_path):
    async def _run():
        state = StateKernel(data_dir=tmp_path)
        async with build_memory_store(data_dir=tmp_path) as store:
            work_channel = InMemoryChannel()
            result_channel = InMemoryChannel()
            dispatcher = WorkDispatcher(
                state_api=state,
                work_sender=work_channel,
                result_receiver=result_channel,
                config=DispatchConfig(worker_count=1, poll_interval=0),
            )
            worker = WorkWorker(
                state_api=state,
                runner=FailingRunner(),
                memory_manager=LangMemMemoryManager(store),
                work_receiver=work_channel,
                result_sender=result_channel,
            )

            task_id = state.task_create("work", {"message": "do"})

            await dispatcher.run_once()
            await worker.run_once()
            await dispatcher.run_once()

            task = state.task_get(task_id)
            assert task is not None
            assert task.state == "failed"

    asyncio.run(_run())
