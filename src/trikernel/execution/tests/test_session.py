import asyncio
import time

from trikernel.execution.session import TrikernelSession
from trikernel.orchestration_kernel.models import RunResult
from trikernel.state_kernel import StateKernel, build_memory_store


class SleepRunner:
    def __init__(self, sleep_seconds: float) -> None:
        self.sleep_seconds = sleep_seconds

    def run(self, task, *, conversation_id: str, stream: bool = False):
        time.sleep(self.sleep_seconds)
        return RunResult(user_output="done", task_state="done")


class FailClaimStateKernel(StateKernel):
    def task_claim(self, filter_by, claimer_id, ttl_seconds):
        return None


class RaisingRunner:
    def run(self, task, *, conversation_id: str, stream: bool = False):
        raise RuntimeError("fail")


def test_main_runner_timeout_marks_failed(tmp_path):
    async def _run():
        state = StateKernel(data_dir=tmp_path)
        async with build_memory_store(data_dir=tmp_path) as store:
            session = TrikernelSession(
                state_api=state,
                runner=SleepRunner(0.2),
                store=store,
                main_runner_timeout_seconds=0.05,
            )
            result = session.send_message("hello")
            assert result.task_state == "failed"
            tasks = state.task_list(task_type="user_request", state=None)
            assert tasks
            assert tasks[0].state == "failed"

    asyncio.run(_run())


def test_claim_failure_marks_failed(tmp_path):
    async def _run():
        state = FailClaimStateKernel(data_dir=tmp_path)
        async with build_memory_store(data_dir=tmp_path) as store:
            session = TrikernelSession(
                state_api=state,
                runner=SleepRunner(0),
                store=store,
            )
            result = session.send_message("hello")
            assert result.task_state == "failed"
            tasks = state.task_list(task_type="user_request", state=None)
            assert tasks
            assert tasks[0].state == "failed"

    asyncio.run(_run())


def test_notification_drain_marks_done(tmp_path):
    async def _run():
        state = StateKernel(data_dir=tmp_path)
        state.task_create("notification", {"message": "ping"})
        async with build_memory_store(data_dir=tmp_path) as store:
            session = TrikernelSession(
                state_api=state,
                runner=SleepRunner(0),
                store=store,
            )
            messages = session.drain_notifications()
            assert messages == ["ping"]
            tasks = state.task_list(task_type="notification", state=None)
            assert tasks[0].state == "done"

    asyncio.run(_run())


def test_main_runner_exception_marks_failed(tmp_path):
    async def _run():
        state = StateKernel(data_dir=tmp_path)
        async with build_memory_store(data_dir=tmp_path) as store:
            session = TrikernelSession(
                state_api=state,
                runner=RaisingRunner(),
                store=store,
            )
            result = session.send_message("hello")
            assert result.task_state == "failed"
            tasks = state.task_list(task_type="user_request", state=None)
            assert tasks
            assert tasks[0].state == "failed"

    asyncio.run(_run())
