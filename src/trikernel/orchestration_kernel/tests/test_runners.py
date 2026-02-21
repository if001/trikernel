from trikernel.orchestration_kernel.models import LLMResponse, RunnerContext
from trikernel.orchestration_kernel.runners import SingleTurnRunner, ToolLoopRunner, PDCARunner
from trikernel.state_kernel.models import Task
from trikernel.tool_kernel.protocols import ToolAPI


class DummyStateAPI:
    def turn_list_recent(self, conversation_id, limit):
        return []


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


class DummyLLM:
    def generate(self, task, tools):
        return LLMResponse(user_output="ok", tool_calls=[])


def _context():
    return RunnerContext(
        runner_id="main",
        state_api=DummyStateAPI(),
        tool_api=DummyToolAPI(),
        llm_api=DummyLLM(),
        tool_llm_api=None,
    )


def test_single_turn_requires_message():
    runner = SingleTurnRunner()
    task = Task(task_id="t1", task_type="user_request", payload={}, state="queued")
    result = runner.run(task, _context())
    assert result.task_state == "failed"
    assert result.error["code"] == "MISSING_MESSAGE"


def test_tool_loop_requires_message():
    runner = ToolLoopRunner()
    task = Task(task_id="t1", task_type="user_request", payload={}, state="queued")
    result = runner.run(task, _context())
    assert result.task_state == "failed"
    assert result.error["code"] == "MISSING_MESSAGE"


def test_pdca_requires_message():
    runner = PDCARunner()
    task = Task(task_id="t1", task_type="user_request", payload={}, state="queued")
    result = runner.run(task, _context())
    assert result.task_state == "failed"
    assert result.error["code"] == "MISSING_MESSAGE"
