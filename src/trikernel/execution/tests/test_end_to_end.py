import asyncio
import time

from trikernel.execution.dispatcher import DispatchConfig, WorkDispatcher
from trikernel.execution.worker import WorkWorker
from trikernel.execution.transports import ResultReceiver, ResultSender, WorkReceiver, WorkSender
from trikernel.state_kernel.kernel import StateKernel
from trikernel.orchestration_kernel.models import RunResult
from trikernel.orchestration_kernel.protocols import Runner
from trikernel.tool_kernel.protocols import ToolAPI


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


class DummyToolAPI(ToolAPI):
    def tool_register(self, tool_definition, handler) -> None:
        return None

    def tool_register_structured(self, tool_definition, tool) -> None:
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


class DummyRunner(Runner):
    def run(self, task, runner_context):
        return RunResult(user_output="done", task_state="done")


class FailingRunner(Runner):
    def run(self, task, runner_context):
        raise RuntimeError("boom")


def test_work_task_end_to_end(tmp_path):
    state = StateKernel(data_dir=tmp_path)
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
        tool_api=DummyToolAPI(),
        runner=DummyRunner(),
        llm_api=None,
        tool_llm_api=None,
        work_receiver=work_channel,
        result_sender=result_channel,
    )

    task_id = state.task_create("work", {"message": "do", "meta": {"channel_id": 1}})

    asyncio.run(dispatcher.run_once())
    asyncio.run(worker.run_once())
    asyncio.run(dispatcher.run_once())

    task = state.task_get(task_id)
    assert task is not None
    assert task.state == "done"
    notifications = state.task_list(task_type="notification", state=None)
    assert notifications
    assert notifications[0].payload.get("meta", {}).get("channel_id") == 1


def test_work_task_failure_marks_failed(tmp_path):
    state = StateKernel(data_dir=tmp_path)
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
        tool_api=DummyToolAPI(),
        runner=FailingRunner(),
        llm_api=None,
        tool_llm_api=None,
        work_receiver=work_channel,
        result_sender=result_channel,
    )

    task_id = state.task_create("work", {"message": "do"})

    asyncio.run(dispatcher.run_once())
    asyncio.run(worker.run_once())
    asyncio.run(dispatcher.run_once())

    task = state.task_get(task_id)
    assert task is not None
    assert task.state == "failed"
