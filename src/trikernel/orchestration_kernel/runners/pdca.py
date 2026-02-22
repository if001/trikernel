from __future__ import annotations

from typing import List

from typing import List, Sequence, Set, Tuple
from langchain_core.messages import BaseMessage, HumanMessage


from trikernel.orchestration_kernel.runners.common import safe_json_load
from trikernel.orchestration_kernel.tool_calls import execute_tool_calls
from trikernel.tool_kernel.structured_tool import TrikernelStructuredTool

from ..logging import get_logger
from .prompts import (
    build_check_step_prompt,
    build_discover_tools_prompt,
    build_do_followup_prompt,
    build_do_step_prompt,
    build_plan_step_prompt,
)
from ..message_utils import (
    ensure_ai_message,
    history_to_messages,
    messages_to_history,
    tool_message_from_result,
)
from ..models import Budget, LLMResponse, RunResult, RunnerContext, StepContext
from ..payloads import build_llm_payload, extract_user_message
from ..protocols import Runner
from ...state_kernel.models import Task


logger = get_logger(__name__)


class PDCARunner(Runner):
    def run(self, task: Task, runner_context: RunnerContext) -> RunResult:
        if not extract_user_message(task.payload or {}):
            return RunResult(
                user_output=None,
                task_state="failed",
                artifact_refs=[],
                error={"code": "MISSING_MESSAGE", "message": "message is required"},
                stream_chunks=[],
            )
        step_context = _initial_step_context(task)
        history = []
        if runner_context.runner_id == "main":
            history = runner_context.state_api.turn_list_recent("default", 5)
        messages: List[BaseMessage] = history_to_messages(history)
        tools = runner_context.tool_api.tool_structured_list()

        while step_context.budget.remaining_steps > 0:
            step_goal, step_success_criteria = plan_step(
                task, step_context, runner_context
            )
            logger.info(f"step_goal {step_goal}")
            logger.info(f"step_success {step_success_criteria}")

            step_toolset = discover_step_tools(
                task,
                step_context,
                step_goal,
                step_success_criteria,
                runner_context,
                tools,
                messages,
            )
            logger.info(f"step_tool set: {step_toolset}")
            do_response, tool_results = do_step(
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
            achieved, evaluation, gaps = check_step(
                task,
                step_context,
                step_goal,
                step_success_criteria,
                do_response,
                runner_context,
            )
            act_step(step_context, step_goal, evaluation, gaps)
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


def _initial_step_context(task: Task) -> StepContext:
    payload = task.payload or {}
    budget_payload = payload.get("budget") or {}
    budget = Budget(
        remaining_steps=int(budget_payload.get("remaining_steps", 10)),
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


def plan_step(
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
    plan = safe_json_load(response.user_output)
    return (
        plan.get("step_goal") or response.user_output or "",
        plan.get("step_success_criteria") or "",
    )


def discover_step_tools(
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
    discover = safe_json_load(response.user_output)
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


def do_step(
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


def check_step(
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
    check = safe_json_load(response.user_output)
    achieved = bool(check.get("achieved", False))
    evaluation = check.get("evaluation") or response.user_output or ""
    gaps = list(check.get("gaps") or [])
    return achieved, evaluation, gaps


def act_step(
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
