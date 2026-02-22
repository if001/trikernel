from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from trikernel.orchestration_kernel.tool_calls import execute_tool_calls
from trikernel.orchestration_kernel.types import ToolResult
from trikernel.tool_kernel.structured_tool import TrikernelStructuredTool

from ..logging import get_logger
from ..message_utils import (
    ensure_ai_message,
    history_to_messages,
    tool_message_from_result,
)
from ..models import Budget, LLMResponse, RunResult, RunnerContext, SimpleStepContext
from ..payloads import build_llm_payload, extract_user_message
from ..protocols import Runner
from ...state_kernel.models import Task
from .common import (
    add_budget_exceeded_message,
)

from .prompts import (
    build_discover_tools_simple_prompt,
    build_tool_loop_followup_prompt,
    build_tool_loop_followup_prompt_for_notification,
    build_tool_loop_followup_prompt_for_worker,
    build_tool_loop_prompt_simple,
    build_tool_loop_prompt_simple_for_notification,
    build_tool_loop_prompt_simple_for_worker,
)


logger = get_logger(__name__)


class ToolLoopRunner(Runner):
    def run(self, task: Task, runner_context: RunnerContext) -> RunResult:
        logger.info(f"role: {task.task_type}")
        step_context = _initial_simple_step_context(task)
        history = []
        if runner_context.runner_id == "main":
            history = runner_context.state_api.turn_list_recent("default", 5)
        history_messages = history_to_messages(history)
        tool_messages: List[BaseMessage] = []
        tools = runner_context.tool_api.tool_structured_list()
        user_message = extract_user_message(task.payload or {})
        if not user_message:
            return RunResult(
                user_output=None,
                task_state="failed",
                artifact_refs=[],
                error={"code": "MISSING_MESSAGE", "message": "message is required"},
                stream_chunks=[],
            )
        completed = False
        while step_context.budget.remaining_steps > 0:
            step_toolset = _discover_step_tools_simple(
                task,
                step_context,
                runner_context,
                user_message,
                history_messages + tool_messages,
            )
            logger.info(f"step_toolset: {step_toolset}")
            response, tool_results = _tool_loop_step(
                task,
                step_context,
                user_message,
                history_messages,
                tool_messages,
                runner_context,
                tools,
                step_toolset,
            )
            if response.tool_calls:
                tool_messages.append(ensure_ai_message(response))
            if tool_results:
                for tool_result in tool_results:
                    tool_messages.append(tool_message_from_result(tool_result))
            step_context.budget.spent_steps += 1
            step_context.budget.remaining_steps -= 1

            if not response.tool_calls:
                completed = True
                break

        if not completed and step_context.budget.remaining_steps <= 0:
            logger.error("Step budget exceeded.")
            add_budget_exceeded_message(tool_messages)

        final_response = _tool_loop_finalize(
            task,
            step_context,
            history_messages,
            tool_messages,
            runner_context,
        )
        budget_error = None
        task_state = "done"
        return RunResult(
            user_output=final_response.user_output,
            task_state=task_state,
            artifact_refs=[],
            error=budget_error,
            stream_chunks=[],
        )


def _initial_simple_step_context(task: Task) -> SimpleStepContext:
    payload = task.payload or {}
    budget_payload = payload.get("budget") or {}
    budget = Budget(
        remaining_steps=int(budget_payload.get("remaining_steps", 10)),
        spent_steps=int(budget_payload.get("spent_steps", 0)),
    )
    _task = "main"
    if task.task_type == "user_request":
        _task = "main"
    else:
        _task = "worker"

    return SimpleStepContext(
        role=_task,
        task_type=task.task_type,
        tool_summary="",
        budget=budget,
    )


def _discover_step_tools_simple(
    task: Task,
    step_context: SimpleStepContext,
    runner_context: RunnerContext,
    user_input: str,
    messages: Sequence[BaseMessage],
) -> Set[str]:
    tools_text = "tool_list:"
    for v in runner_context.tool_api.tool_descriptions():
        tools_text += f"{v['tool_name']}: {v['description']}\n"

    prompt = build_discover_tools_simple_prompt(
        user_input=user_input,
        tools_text=tools_text,
        step_context_text=step_context.to_str(),
    )
    discover_messages = list(messages) + [HumanMessage(content=prompt)]
    discover_task = Task(
        task_id=task.task_id,
        task_type="pdca.discover",
        payload=build_llm_payload(messages=discover_messages),
        state="running",
    )

    response = runner_context.llm_api.generate(discover_task, [])
    query = response.user_output
    logger.info(f"tool query: {query}")
    selected = runner_context.tool_api.tool_search(str(query))
    logger.info(f"selected: {selected}")
    return set(selected)


def _tool_loop_step(
    task: Task,
    step_context: SimpleStepContext,
    user_message: str,
    base_messages: List[BaseMessage],
    tool_messages: List[BaseMessage],
    runner_context: RunnerContext,
    tools: Sequence[TrikernelStructuredTool],
    step_toolset: Set[str],
) -> Tuple[LLMResponse, List[ToolResult]]:
    allowed_tools = [tool for tool in tools if tool.name in step_toolset]

    if task.task_type == "user_request":
        prompt = build_tool_loop_prompt_simple(
            user_message=user_message,
            step_context_text=step_context.to_str(),
        )
    elif task.task_type == "notification":
        prompt = build_tool_loop_prompt_simple_for_notification(
            message=user_message,
            step_context_text=step_context.to_str(),
        )
    elif task.task_type == "work":
        prompt = build_tool_loop_prompt_simple_for_worker(
            message=user_message,
            step_context_text=step_context.to_str(),
        )
    else:
        logger.error(f"bad task type selected: {task.task_type}")
        prompt = build_tool_loop_prompt_simple(
            user_message=user_message,
            step_context_text=step_context.to_str(),
        )

    messages = (
        list(base_messages) + [HumanMessage(content=prompt)] + list(tool_messages)
    )
    do_task = Task(
        task_id=task.task_id,
        task_type="tool_loop.step",
        payload=build_llm_payload(messages=messages),
        state="running",
    )
    response = runner_context.llm_api.generate(do_task, allowed_tools)
    tool_results = execute_tool_calls(
        runner_context, task, response.tool_calls, allowed_tools=step_toolset
    )
    return response, tool_results


def _tool_loop_finalize(
    task: Task,
    step_context: SimpleStepContext,
    base_messages: List[BaseMessage],
    tool_messages: List[BaseMessage],
    runner_context: RunnerContext,
) -> LLMResponse:
    payload = task.payload or {}
    user_message = extract_user_message(payload)

    if task.task_type == "user_request":
        final_prompt = build_tool_loop_followup_prompt(
            user_message=user_message,
            step_context_text=step_context.to_str(),
        )
    elif task.task_type == "notification":
        final_prompt = build_tool_loop_followup_prompt_for_notification(
            message=user_message,
            step_context_text=step_context.to_str(),
        )
    elif task.task_type == "work":
        final_prompt = build_tool_loop_followup_prompt_for_worker(
            message=user_message,
            step_context_text=step_context.to_str(),
        )
    else:
        logger.error(f"bad task type selected: {task.task_type}")
        final_prompt = build_tool_loop_followup_prompt(
            user_message=user_message,
            step_context_text=step_context.to_str(),
        )
    messages = (
        list(base_messages) + [HumanMessage(content=final_prompt)] + list(tool_messages)
    )
    logger.info(f"final_input: {messages}")
    final_task = Task(
        task_id=task.task_id,
        task_type="tool_loop.final",
        payload=build_llm_payload(messages=messages),
        state="running",
    )
    return runner_context.llm_api.generate(final_task, [])
