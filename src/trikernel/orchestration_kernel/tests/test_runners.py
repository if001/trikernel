from trikernel.orchestration_kernel.models import LLMResponse, RunnerContext
from trikernel.orchestration_kernel.runners import LangGraphToolLoopRunner
from trikernel.state_kernel.models import Task
from trikernel.tool_kernel.protocols import ToolLLMBase


class DummyStateAPI:
    def memory_kernel(self, conversation_id: str):
        return None


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


class DummyMessageStore:
    checkpointer = None


class DummyStore:
    def search(self, *args, **kwargs):
        return []


class DummyLLM:
    def generate(self, task, tools):
        return LLMResponse(user_output="ok", tool_calls=[])

    def collect_stream(self, task, tools):
        return LLMResponse(user_output="ok", tool_calls=[]), []


class DummyToolLLM(ToolLLMBase):
    def generate(self, prompt: str, tools=None) -> str:
        return ""


def _context():
    return RunnerContext(
        runner_id="main",
        conversation_id="default",
        state_api=DummyStateAPI(),
        message_store=DummyMessageStore(),
        tool_api=DummyToolAPI(),
        llm_api=DummyLLM(),
        tool_llm_api=DummyToolLLM(),
        store=DummyStore(),
    )


def test_langgraph_tool_loop_requires_message():
    runner = LangGraphToolLoopRunner()
    task = Task(task_id="t1", task_type="user_request", payload={}, state="queued")
    result = runner.run(task, _context())
    assert result.task_state == "failed"
    assert result.error["code"] == "MISSING_MESSAGE"
