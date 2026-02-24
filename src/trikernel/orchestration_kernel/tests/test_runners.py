from trikernel.orchestration_kernel.models import LLMResponse, RunnerContext
from trikernel.orchestration_kernel.runners import LangGraphToolLoopRunner
from trikernel.state_kernel.models import Task


class DummyStateAPI:
    pass


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


class DummyLLM:
    def generate(self, task, tools):
        return LLMResponse(user_output="ok", tool_calls=[])


def _context():
    return RunnerContext(
        runner_id="main",
        conversation_id="default",
        state_api=DummyStateAPI(),
        message_store=DummyMessageStore(),
        tool_api=DummyToolAPI(),
        llm_api=DummyLLM(),
        tool_llm_api=None,
    )


def test_langgraph_tool_loop_requires_message():
    runner = LangGraphToolLoopRunner()
    task = Task(task_id="t1", task_type="user_request", payload={}, state="queued")
    result = runner.run(task, _context())
    assert result.task_state == "failed"
    assert result.error["code"] == "MISSING_MESSAGE"
