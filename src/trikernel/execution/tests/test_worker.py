import asyncio

from trikernel.execution.worker import WorkWorker
from trikernel.execution.transports import ResultSender, WorkReceiver
from trikernel.state_kernel import LangMemMemoryManager, StateKernel, build_memory_store
from trikernel.orchestration_kernel.models import RunResult


class FakeWorkReceiver(WorkReceiver):
    def __init__(self, payload):
        self._payload = payload
        self._used = False

    async def recv_json(self):
        if self._used:
            raise Exception("empty")
        self._used = True
        return self._payload

    async def try_recv_json(self):
        if self._used:
            return None
        self._used = True
        return self._payload


class FakeResultSender(ResultSender):
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class DummyRunner:
    def run(self, task, *, conversation_id: str, stream: bool = False):
        return RunResult(user_output="done", task_state="done")


def test_worker_sends_result(tmp_path):
    async def _run():
        state = StateKernel(data_dir=tmp_path)
        async with build_memory_store(data_dir=tmp_path) as store:
            task_id = state.task_create("work", {"message": "do"})
            receiver = FakeWorkReceiver({"task_id": task_id})
            sender = FakeResultSender()
            worker = WorkWorker(
                state_api=state,
                runner=DummyRunner(),
                memory_manager=LangMemMemoryManager(store),
                work_receiver=receiver,
                result_sender=sender,
            )
            await worker.run_once()
            assert sender.sent
            assert sender.sent[0]["task_id"] == task_id
            assert sender.sent[0]["task_state"] == "done"

    asyncio.run(_run())
