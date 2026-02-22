from __future__ import annotations

from typing import List

from langchain_core.messages import BaseMessage, HumanMessage

from ..logging import get_logger
from ..message_utils import history_to_messages
from ..models import RunResult, RunnerContext
from ..payloads import build_llm_payload, extract_user_message
from ..protocols import Runner
from ...state_kernel.models import Task

logger = get_logger(__name__)


class SingleTurnRunner(Runner):
    def run(self, task: Task, runner_context: RunnerContext) -> RunResult:
        user_message = extract_user_message(task.payload or {})
        if not user_message:
            return RunResult(
                user_output=None,
                task_state="failed",
                artifact_refs=[],
                error={"code": "MISSING_MESSAGE", "message": "message is required"},
                stream_chunks=[],
            )
        messages: List[BaseMessage] = []
        if runner_context.runner_id == "main":
            recent = runner_context.state_api.turn_list_recent("default", 5)
            messages.extend(history_to_messages(recent))
        messages.append(HumanMessage(content=user_message))
        llm_task = Task(
            task_id=task.task_id,
            task_type=task.task_type,
            payload=build_llm_payload(messages=messages),
            state="running",
        )
        tools = runner_context.tool_api.tool_structured_list()
        stream_chunks: List[str] = []
        if runner_context.stream and hasattr(runner_context.llm_api, "collect_stream"):
            response, stream_chunks = runner_context.llm_api.collect_stream(
                llm_task, tools
            )
        else:
            response = runner_context.llm_api.generate(llm_task, tools)

        tool_results = []
        if response.tool_calls:
            from ..tool_calls import execute_tool_calls

            tool_results = execute_tool_calls(
                runner_context, task, response.tool_calls, allowed_tools=None
            )

        user_output = response.user_output
        if user_output is None and tool_results:
            user_output = f"Tool results: {tool_results}"

        return RunResult(
            user_output=user_output,
            task_state="done",
            artifact_refs=[],
            error=None,
            stream_chunks=stream_chunks,
        )
