import time

from trikernel.execution.session import TrikernelSession
from trikernel.orchestration_kernel.models import RunResult
from trikernel.orchestration_kernel.protocols import Runner
from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.protocols import ToolAPI


class DummyToolAPI(ToolAPI):
    def tool_register(self, tool_definition, handler) -> None:
        return None

    def tool_register_structured(self, tool) -> None:
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


class SleepRunner(Runner):
    def __init__(self, sleep_seconds: float) -> None:
        self.sleep_seconds = sleep_seconds

    def run(self, task, runner_context):
        time.sleep(self.sleep_seconds)
        return RunResult(user_output="done", task_state="done")


class DummyLLM:
    def generate(self, task, tools):
        raise NotImplementedError

    def collect_stream(self, task, tools):
        raise NotImplementedError


def test_main_runner_timeout_marks_failed(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    session = TrikernelSession(
        state_api=state,
        tool_api=DummyToolAPI(),
        runner=SleepRunner(0.2),
        llm_api=DummyLLM(),
        tool_llm_api=None,
        main_runner_timeout_seconds=0.05,
    )
    result = session.send_message("hello")
    assert result.task_state == "failed"
    tasks = state.task_list(task_type="user_request", state=None)
    assert tasks
    assert tasks[0].state == "failed"


class FailClaimStateKernel(StateKernel):
    def task_claim(self, filter_by, claimer_id, ttl_seconds):
        return None


class RaisingRunner(Runner):
    def run(self, task, runner_context):
        raise RuntimeError("fail")


def test_claim_failure_marks_failed(tmp_path):
    state = FailClaimStateKernel(data_dir=tmp_path)
    session = TrikernelSession(
        state_api=state,
        tool_api=DummyToolAPI(),
        runner=SleepRunner(0),
        llm_api=DummyLLM(),
        tool_llm_api=None,
    )
    result = session.send_message("hello")
    assert result.task_state == "failed"
    tasks = state.task_list(task_type="user_request", state=None)
    assert tasks
    assert tasks[0].state == "failed"


def test_notification_drain_marks_done(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    state.task_create("notification", {"message": "ping"})
    session = TrikernelSession(
        state_api=state,
        tool_api=DummyToolAPI(),
        runner=SleepRunner(0),
        llm_api=DummyLLM(),
        tool_llm_api=None,
    )
    messages = session.drain_notifications()
    assert messages == ["ping"]
    tasks = state.task_list(task_type="notification", state=None)
    assert tasks[0].state == "done"


def test_main_runner_exception_marks_failed(tmp_path):
    state = StateKernel(data_dir=tmp_path)
    session = TrikernelSession(
        state_api=state,
        tool_api=DummyToolAPI(),
        runner=RaisingRunner(),
        llm_api=DummyLLM(),
        tool_llm_api=None,
    )
    result = session.send_message("hello")
    assert result.task_state == "failed"
    tasks = state.task_list(task_type="user_request", state=None)
    assert tasks
    assert tasks[0].state == "failed"
