from langchain_core.messages import AIMessage

from trikernel.orchestration_kernel.runners.simple_tool_loop import (
    SimpleGraphToolLoopRunner,
)
from trikernel.state_kernel.models import Task
from trikernel.tool_kernel.protocols import ToolLLMBase


class DummyStateAPI:
    def memory_kernel(self, conversation_id: str):
        return None


class DummyToolAPI:
    def __init__(self) -> None:
        self._tool_llm = DummyToolLLM()

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

    def tool_llm_api(self):
        return self._tool_llm


class DummyMessageStore:
    checkpointer = None


class DummyStore:
    def search(self, *args, **kwargs):
        return []


class DummyChatModel:
    def invoke(self, *args, **kwargs):
        return AIMessage(content="")

    def bind_tools(self, *args, **kwargs):
        return self


class DummyToolLLM(ToolLLMBase):
    def generate(self, prompt: str, tools=None) -> str:
        return ""


def _runner() -> SimpleGraphToolLoopRunner:
    dummy_llm = DummyChatModel()
    return SimpleGraphToolLoopRunner(
        state_api=DummyStateAPI(),
        tool_api=DummyToolAPI(),
        message_store=DummyMessageStore(),
        store=DummyStore(),
        llm_api=dummy_llm,
        large_llm_api=dummy_llm,
    )


def test_simple_tool_loop_requires_message():
    runner = _runner()
    task = Task(task_id="t1", task_type="user_request", payload={}, state="queued")
    result = runner.run(task, conversation_id="default")
    assert result.task_state == "failed"
    assert result.error["code"] == "MISSING_MESSAGE"
