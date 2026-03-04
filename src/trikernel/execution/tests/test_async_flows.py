import asyncio
import queue
import threading
import time

from trikernel.execution.dispatcher import DispatchConfig, WorkDispatcher
from trikernel.execution.worker import WorkWorker
from trikernel.execution.session import TrikernelSession
from trikernel.execution.transports import (
    ResultReceiver,
    ResultSender,
    WorkReceiver,
    WorkSender,
)
from trikernel.orchestration_kernel.models import RunResult
from trikernel.state_kernel import LangMemMemoryManager, StateKernel, build_memory_store


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


class StaticRunner:
    def __init__(self, output: str) -> None:
        self._output = output

    def run(self, task, *, conversation_id: str, stream: bool = False):
        return RunResult(user_output=self._output, task_state="done")


class BlockingRunner:
    def __init__(self, event: threading.Event, output: str) -> None:
        self._event = event
        self._output = output

    def run(self, task, *, conversation_id: str, stream: bool = False):
        self._event.wait()
        return RunResult(user_output=self._output, task_state="done")


def _run_async(coro) -> None:
    asyncio.run(coro)


def test_async_flow_main_worker_serial(tmp_path):
    async def _run():
        state = StateKernel(data_dir=tmp_path)
        async with build_memory_store(data_dir=tmp_path) as store:
            main_runner = StaticRunner("main done")
            worker_runner = StaticRunner("work done")
            session = TrikernelSession(
                state_api=state,
                runner=main_runner,
                store=store,
                enable_memory_updates=False,
            )

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
                runner=worker_runner,
                memory_manager=LangMemMemoryManager(store),
                work_receiver=work_channel,
                result_sender=result_channel,
            )

            result = session.send_message("hello")
            assert result.message == "main done"

            session.create_work_task({"message": "do"})

            await dispatcher.run_once()
            await worker.run_once()
            await dispatcher.run_once()

            notices = session.drain_notifications()
            assert "work done" in notices

    asyncio.run(_run())


def test_async_flow_main_and_worker_parallel(tmp_path):
    async def _run():
        state = StateKernel(data_dir=tmp_path)
        async with build_memory_store(data_dir=tmp_path) as store:
            main_runner = StaticRunner("main done")
            release = threading.Event()
            worker_runner = BlockingRunner(release, "work done")
            session = TrikernelSession(
                state_api=state,
                runner=main_runner,
                store=store,
                enable_memory_updates=False,
            )

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
                runner=worker_runner,
                memory_manager=LangMemMemoryManager(store),
                work_receiver=work_channel,
                result_sender=result_channel,
            )

            session.create_work_task({"message": "do"})
            await dispatcher.run_once()

            worker_thread = threading.Thread(
                target=_run_async, args=(worker.run_once(),)
            )
            worker_thread.start()

            start = time.monotonic()
            result = session.send_message("hello")
            elapsed = time.monotonic() - start
            assert result.message == "main done"
            assert elapsed < 0.5

            release.set()
            worker_thread.join(timeout=2)
            await dispatcher.run_once()

            notices = session.drain_notifications()
            assert "work done" in notices

    asyncio.run(_run())
