from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Set, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from .models import Budget, LLMResponse, RunResult, RunnerContext, StepContext
from .protocols import Runner
from ..state_kernel.models import Task
from ..tool_kernel.structured_tool import TrikernelStructuredTool
from .logging import get_logger
from .message_utils import (
    ensure_ai_message,
    history_to_messages,
    messages_to_history,
    tool_message_from_result,
)
from .payloads import build_llm_payload, extract_user_message
from .prompts import (
    build_check_step_prompt,
    build_discover_tools_prompt,
    build_do_followup_prompt,
    build_do_step_prompt,
    build_plan_step_prompt,
    build_tool_loop_followup_prompt,
    build_tool_loop_prompt,
)
from .tool_calls import execute_tool_calls
from .types import ToolResult

logger = get_logger(__name__)


class SingleTurnRunner(Runner):
    def run(self, task: Task, runner_context: RunnerContext) -> RunResult:
        user_message = extract_user_message(task.payload or {})
        if runner_context.runner_id == "main":
            recent = runner_context.state_api.turn_list_recent("default", 5)
            llm_task = Task(
                task_id=task.task_id,
                task_type=task.task_type,
                payload=build_llm_payload(message=user_message, history=recent),
                state="running",
            )
        else:
            llm_task = Task(
                task_id=task.task_id,
                task_type=task.task_type,
                payload=build_llm_payload(message=user_message),
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


class PDCARunner(Runner):
    def run(self, task: Task, runner_context: RunnerContext) -> RunResult:
        step_context = _initial_step_context(task)
        history = []
        if runner_context.runner_id == "main":
            history = runner_context.state_api.turn_list_recent("default", 5)
        messages = history_to_messages(history)
        tools = runner_context.tool_api.tool_structured_list()
        for v in tools:
            print(v.name)
        while step_context.budget.remaining_steps > 0:
            step_goal, step_success_criteria = _plan_step(
                task, step_context, runner_context
            )
            logger.info(f"step_goal {step_goal}")
            logger.info(f"step_success {step_success_criteria}")

            step_toolset = _discover_step_tools(
                task,
                step_context,
                step_goal,
                step_success_criteria,
                runner_context,
                tools,
                messages,
            )
            logger.info(f"step_tool set: {step_toolset}")
            do_response, tool_results = _do_step(
                task,
                step_context,
                step_goal,
                step_success_criteria,
                runner_context,
                tools,
                step_toolset,
            )
            logger.info(f"do_resp, {do_response}")
            logger.info(f"tool_res, {tool_results}")
            achieved, evaluation, gaps = _check_step(
                task,
                step_context,
                step_goal,
                step_success_criteria,
                do_response,
                runner_context,
            )
            _act_step(step_context, step_goal, evaluation, gaps)
            step_context.budget.spent_steps += 1
            step_context.budget.remaining_steps -= 1
            logger.info(f"achieved {achieved}")
            if achieved:
                final_message = do_response.user_output or evaluation
                if runner_context.runner_id == "worker":
                    runner_context.state_api.task_create(
                        "notification",
                        {
                            "message": final_message,
                            "severity": "info",
                            "related_task_id": task.task_id,
                        },
                    )
                    final_message = None
                return RunResult(
                    user_output=final_message,
                    task_state="done",
                    artifact_refs=[],
                    error=None,
                    stream_chunks=[],
                )

        return RunResult(
            user_output=None,
            task_state="failed",
            artifact_refs=[],
            error={"code": "BUDGET_EXCEEDED", "message": "Step budget exceeded."},
            stream_chunks=[],
        )


class ToolLoopRunner(Runner):
    def run(self, task: Task, runner_context: RunnerContext) -> RunResult:
        step_context = _initial_step_context(task)
        history = []
        if runner_context.runner_id == "main":
            history = runner_context.state_api.turn_list_recent("default", 5)
        base_messages = history_to_messages(history)
        tool_messages: List[BaseMessage] = []
        tools = runner_context.tool_api.tool_structured_list()
        completed = False
        while step_context.budget.remaining_steps > 0:
            step_toolset = _discover_step_tools(
                task,
                step_context,
                "",
                "",
                runner_context,
                tools,
                base_messages + tool_messages,
                enforce_search=True,
            )
            logger.info(f"step_toolset: {step_toolset}")
            response, tool_results = _tool_loop_step(
                task,
                step_context,
                base_messages,
                tool_messages,
                runner_context,
                tools,
                step_toolset,
            )
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
            tool_messages.append(
                ToolMessage(
                    content='{"error":"Step budget exceeded.","code":"BUDGET_EXCEEDED"}',
                    tool_call_id="system",
                )
            )

        final_response = _tool_loop_finalize(
            task,
            step_context,
            base_messages,
            tool_messages,
            runner_context,
        )
        budget_error = None
        task_state = "done"
        if not completed and step_context.budget.remaining_steps <= 0:
            budget_error = {
                "code": "BUDGET_EXCEEDED",
                "message": "Step budget exceeded.",
            }
            task_state = "failed"
        return RunResult(
            user_output=final_response.user_output,
            task_state=task_state,
            artifact_refs=[],
            error=budget_error,
            stream_chunks=[],
        )


def _initial_step_context(task: Task) -> StepContext:
    payload = task.payload or {}
    budget_payload = payload.get("budget") or {}
    budget = Budget(
        remaining_steps=int(budget_payload.get("remaining_steps", 3)),
        spent_steps=int(budget_payload.get("spent_steps", 0)),
    )
    context_payload = payload.get("step_context") or {}
    return StepContext(
        facts=list(context_payload.get("facts", [])),
        open_issues=list(context_payload.get("open_issues", [])),
        plan=list(context_payload.get("plan", [])),
        last_result=context_payload.get("last_result", ""),
        budget=budget,
    )


def _plan_step(
    task: Task, step_context: StepContext, runner_context: RunnerContext
) -> Tuple[str, str]:
    history = runner_context.state_api.turn_list_recent("default", 5)
    prompt = build_plan_step_prompt(
        task_payload=task.payload or {},
        step_context=step_context.to_dict(),
        history=[t.to_dict() for t in history],
    )
    plan_task = Task(
        task_id=task.task_id,
        task_type="pdca.plan",
        payload=build_llm_payload(message=prompt),
        state="running",
    )
    response = runner_context.llm_api.generate(plan_task, [])
    plan = _safe_json_load(response.user_output)
    step_goal = plan.get("step_goal") or response.user_output or "Plan step"
    step_success_criteria = plan.get("step_success_criteria") or ""
    return step_goal, step_success_criteria


def _do_step(
    task: Task,
    step_context: StepContext,
    step_goal: str,
    step_success_criteria: str,
    runner_context: RunnerContext,
    tools: Sequence[TrikernelStructuredTool],
    step_toolset: Set[str],
) -> Tuple[LLMResponse, List[ToolResult]]:
    allowed_tools = [tool for tool in tools if tool.name in step_toolset]
    prompt = build_do_step_prompt(
        step_goal=step_goal,
        step_success_criteria=step_success_criteria,
        step_context=step_context.to_dict(),
        step_toolset=sorted(step_toolset),
    )
    messages: List[BaseMessage] = [HumanMessage(content=prompt)]
    do_task = Task(
        task_id=task.task_id,
        task_type="pdca.do",
        payload=build_llm_payload(messages=messages),
        state="running",
    )
    response = runner_context.llm_api.generate(do_task, allowed_tools)
    messages.append(ensure_ai_message(response))
    tool_results = execute_tool_calls(
        runner_context, task, response.tool_calls, allowed_tools=step_toolset
    )

    if not response.tool_calls:
        return response, tool_results

    for tool_result in tool_results:
        messages.append(tool_message_from_result(tool_result))
    followup_prompt = build_do_followup_prompt(
        step_goal=step_goal,
        step_success_criteria=step_success_criteria,
        step_context=step_context.to_dict(),
    )
    messages.append(HumanMessage(content=followup_prompt))
    followup_task = Task(
        task_id=task.task_id,
        task_type="pdca.do.followup",
        payload=build_llm_payload(messages=messages),
        state="running",
    )
    followup_response = runner_context.llm_api.generate(followup_task, [])
    messages.append(ensure_ai_message(followup_response))
    return followup_response, tool_results


def _tool_loop_step(
    task: Task,
    step_context: StepContext,
    base_messages: List[BaseMessage],
    tool_messages: List[BaseMessage],
    runner_context: RunnerContext,
    tools: Sequence[TrikernelStructuredTool],
    step_toolset: Set[str],
) -> Tuple[LLMResponse, List[ToolResult]]:
    payload = task.payload or {}
    user_message = extract_user_message(payload)
    allowed_tools = [tool for tool in tools if tool.name in step_toolset]
    prompt = build_tool_loop_prompt(
        user_message=user_message,
        step_context=step_context.to_dict(),
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
    step_context: StepContext,
    base_messages: List[BaseMessage],
    tool_messages: List[BaseMessage],
    runner_context: RunnerContext,
) -> LLMResponse:
    payload = task.payload or {}
    user_message = extract_user_message(payload)
    final_prompt = build_tool_loop_followup_prompt(
        user_message=user_message,
        step_context=step_context.to_dict(),
    )
    messages = (
        list(base_messages) + [HumanMessage(content=final_prompt)] + list(tool_messages)
    )
    final_task = Task(
        task_id=task.task_id,
        task_type="tool_loop.final",
        payload=build_llm_payload(messages=messages),
        state="running",
    )
    return runner_context.llm_api.generate(final_task, [])


def _check_step(
    task: Task,
    step_context: StepContext,
    step_goal: str,
    step_success_criteria: str,
    do_response: LLMResponse,
    runner_context: RunnerContext,
) -> Tuple[bool, str, List[str]]:
    user_output = (do_response.user_output or "").strip()
    if not user_output:
        return False, "empty_output", ["empty_output"]
    prompt = build_check_step_prompt(
        step_goal=step_goal,
        step_success_criteria=step_success_criteria,
        step_context=step_context.to_dict(),
        user_output=user_output,
    )
    check_task = Task(
        task_id=task.task_id,
        task_type="pdca.check",
        payload=build_llm_payload(message=prompt),
        state="running",
    )
    response = runner_context.llm_api.generate(check_task, [])
    check = _safe_json_load(response.user_output)
    achieved = bool(check.get("achieved", False))
    evaluation = check.get("evaluation") or response.user_output or ""
    gaps = list(check.get("gaps") or [])
    return achieved, evaluation, gaps


def _act_step(
    step_context: StepContext,
    step_goal: str,
    evaluation: str,
    gaps: List[str],
) -> None:
    step_context.last_result = evaluation
    step_context.open_issues = gaps
    step_context.plan = gaps if gaps else [step_goal]
    if evaluation and evaluation not in step_context.facts:
        step_context.facts.append(evaluation)


def _safe_json_load(text: Optional[str]) -> Dict[str, object]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _discover_step_tools(
    task: Task,
    step_context: StepContext,
    step_goal: str,
    step_success_criteria: str,
    runner_context: RunnerContext,
    tools: Sequence[TrikernelStructuredTool],
    messages: Sequence[BaseMessage],
    enforce_search: bool = False,
) -> Set[str]:
    prompt = build_discover_tools_prompt(
        step_goal=step_goal,
        step_success_criteria=step_success_criteria,
        step_context=step_context.to_dict(),
        history=messages_to_history(messages),
    )
    discover_task = Task(
        task_id=task.task_id,
        task_type="pdca.discover",
        payload=build_llm_payload(message=prompt),
        state="running",
    )
    response = runner_context.llm_api.generate(discover_task, tools)
    discover = _safe_json_load(response.user_output)
    queries = list(discover.get("search_queries") or [])
    if enforce_search and not queries:
        queries = [step_goal]
    searched: Set[str] = set()
    for query in queries:
        searched.update(runner_context.tool_api.tool_search(str(query)))
    described = [
        runner_context.tool_api.tool_describe(name).tool_name for name in searched
    ]
    selected = set(discover.get("selected_tools") or [])
    if not selected:
        selected = set(described)
    return selected
