import asyncio

from trikernel.execution.worker import WorkWorker
from trikernel.execution.transports import ResultSender, WorkReceiver
from trikernel.state_kernel.kernel import StateKernel
from trikernel.state_kernel.message_store import build_message_store
from trikernel.state_kernel.memory_store import build_memory_store
from trikernel.orchestration_kernel.models import LLMResponse, RunResult
from trikernel.state_kernel.memory_manager import LangMemMemoryManager
from trikernel.tool_kernel.protocols import ToolLLMBase


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


class DummyToolAPI:
    def tool_register(self, tool) -> None:
        return None

    def tool_describe(self, tool_name):
        raise KeyError(tool_name)

    def tool_search(self, query):
        return []

    def tool_list(self):
        return []

    def tool_descriptions(self):
        return []

    def tool_structured_list(self):
        return []


class DummyLLM:
    def generate(self, task, tools):
        return LLMResponse(user_output="ok", tool_calls=[])

    def collect_stream(self, task, tools):
        return LLMResponse(user_output="ok", tool_calls=[]), []


class DummyToolLLM(ToolLLMBase):
    def generate(self, prompt: str, tools=None) -> str:
        return ""


class DummyRunner:
    def run(self, task, runner_context):
        return RunResult(user_output="done", task_state="done")


def test_worker_sends_result(tmp_path):
    async def _run():
        state = StateKernel(data_dir=tmp_path)
        async with build_memory_store(data_dir=tmp_path) as store, build_message_store(
            data_dir=tmp_path
        ) as message_store:
            task_id = state.task_create("work", {"message": "do"})
            receiver = FakeWorkReceiver({"task_id": task_id})
            sender = FakeResultSender()
            worker = WorkWorker(
                state_api=state,
                message_store=message_store,
                tool_api=DummyToolAPI(),
                runner=DummyRunner(),
                llm_api=DummyLLM(),
                tool_llm_api=DummyToolLLM(),
                memory_manager=LangMemMemoryManager(store),
                store=store,
                work_receiver=receiver,
                result_sender=sender,
            )
            await worker.run_once()
            assert sender.sent
            assert sender.sent[0]["task_id"] == task_id
            assert sender.sent[0]["task_state"] == "done"

    asyncio.run(_run())
