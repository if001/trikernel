import asyncio
import queue
import threading
import time

from trikernel.execution.dispatcher import DispatchConfig, PendingWork, WorkDispatcher
from trikernel.execution.worker import WorkWorker
from trikernel.execution.session import TrikernelSession
from trikernel.execution.transports import ResultReceiver, ResultSender, WorkReceiver, WorkSender
from trikernel.orchestration_kernel.models import RunResult
from trikernel.orchestration_kernel.protocols import Runner
from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.protocols import ToolAPI


class ThreadSafeChannel(WorkSender, WorkReceiver, ResultSender, ResultReceiver):
    def __init__(self) -> None:
        self._queue: queue.Queue[dict] = queue.Queue()

    async def send_json(self, payload):
        self._queue.put(payload)

    async def recv_json(self):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._queue.get)

    async def try_recv_json(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


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


class StaticRunner(Runner):
    def __init__(self, output: str) -> None:
        self._output = output

    def run(self, task, runner_context):
        return RunResult(user_output=self._output, task_state="done")


class BlockingRunner(Runner):
    def __init__(self, event: threading.Event, output: str) -> None:
        self._event = event
        self._output = output

    def run(self, task, runner_context):
        self._event.wait()
        return RunResult(user_output=self._output, task_state="done")


def _run_async(coro) -> None:
    asyncio.run(coro)


def test_async_flow_main_worker_serial(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    tool_api = DummyToolAPI()
    main_runner = StaticRunner("main done")
    worker_runner = StaticRunner("work done")
    session = TrikernelSession(state, tool_api, main_runner, llm_api=None)

    work_channel = ThreadSafeChannel()
    result_channel = ThreadSafeChannel()
    dispatcher = WorkDispatcher(
        state_api=state,
        work_sender=work_channel,
        result_receiver=result_channel,
        config=DispatchConfig(worker_count=1, poll_interval=0),
    )
    worker = WorkWorker(
        state_api=state,
        tool_api=tool_api,
        runner=worker_runner,
        llm_api=None,
        tool_llm_api=None,
        work_receiver=work_channel,
        result_sender=result_channel,
    )

    result = session.send_message("hello")
    assert result.message == "main done"

    session.create_work_task({"message": "do"})

    _run_async(dispatcher.run_once())
    _run_async(worker.run_once())
    _run_async(dispatcher.run_once())

    notices = session.drain_notifications()
    assert "work done" in notices


def test_async_flow_main_and_worker_parallel(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    tool_api = DummyToolAPI()
    main_runner = StaticRunner("main done")
    release = threading.Event()
    worker_runner = BlockingRunner(release, "work done")
    session = TrikernelSession(state, tool_api, main_runner, llm_api=None)

    work_channel = ThreadSafeChannel()
    result_channel = ThreadSafeChannel()
    dispatcher = WorkDispatcher(
        state_api=state,
        work_sender=work_channel,
        result_receiver=result_channel,
        config=DispatchConfig(worker_count=1, poll_interval=0),
    )
    worker = WorkWorker(
        state_api=state,
        tool_api=tool_api,
        runner=worker_runner,
        llm_api=None,
        tool_llm_api=None,
        work_receiver=work_channel,
        result_sender=result_channel,
    )

    session.create_work_task({"message": "do"})
    _run_async(dispatcher.run_once())

    worker_thread = threading.Thread(target=_run_async, args=(worker.run_once(),))
    worker_thread.start()

    start = time.monotonic()
    result = session.send_message("hello")
    elapsed = time.monotonic() - start
    assert result.message == "main done"
    assert elapsed < 0.5

    release.set()
    worker_thread.join(timeout=2)
    _run_async(dispatcher.run_once())

    notices = session.drain_notifications()
    assert "work done" in notices


def test_worker_max_parallel_two(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    dispatcher = WorkDispatcher(
        state_api=state,
        work_sender=ThreadSafeChannel(),
        result_receiver=ThreadSafeChannel(),
        config=DispatchConfig(worker_count=2, poll_interval=0),
    )

    state.task_create("work", {"message": "one"})
    state.task_create("work", {"message": "two"})
    state.task_create("work", {"message": "three"})

    _run_async(dispatcher.run_once())

    assert len(dispatcher._inflight) == 2
    assert len(dispatcher._pending) == 1


def test_worker_timeout_flow(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    dispatcher = WorkDispatcher(
        state_api=state,
        work_sender=ThreadSafeChannel(),
        result_receiver=ThreadSafeChannel(),
        config=DispatchConfig(worker_timeout_seconds=1),
    )
    task_id = state.task_create("work", {"message": "do"})
    dispatcher._inflight[task_id] = time.monotonic() - 5
    dispatcher._fail_timed_out_tasks()
    task = state.task_get(task_id)
    assert task is not None
    assert task.state == "failed"


def test_queue_timeout_flow(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    dispatcher = WorkDispatcher(
        state_api=state,
        work_sender=ThreadSafeChannel(),
        result_receiver=ThreadSafeChannel(),
        config=DispatchConfig(work_queue_timeout_seconds=1),
    )
    task_id = state.task_create("work", {"message": "do"})
    dispatcher._pending.append(
        PendingWork(task_id=task_id, enqueued_at=time.monotonic() - 5, timeout_seconds=1)
    )
    dispatcher._fail_timed_out_pending()
    task = state.task_get(task_id)
    assert task is not None
    assert task.state == "failed"
