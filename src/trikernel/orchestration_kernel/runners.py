from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from .models import Budget, LLMResponse, RunResult, RunnerContext, StepContext
from .protocols import Runner
from ..state_kernel.models import Task, utc_now
from ..tool_kernel.models import ToolContext
from .logging import get_logger
from .prompts import (
    build_check_step_prompt,
    build_discover_tools_prompt,
    build_do_followup_prompt,
    build_do_step_prompt,
    build_plan_step_prompt,
    build_tool_loop_followup_prompt,
    build_tool_loop_prompt,
)

logger = get_logger(__name__)


def _build_tool_context(runner_context: RunnerContext, task: Task) -> ToolContext:
    return ToolContext(
        runner_id=runner_context.runner_id,
        task_id=task.task_id,
        state_api=runner_context.state_api,
        now=utc_now(),
        llm_api=runner_context.tool_llm_api,
    )


class SingleTurnRunner(Runner):
    def run(self, task: Task, runner_context: RunnerContext) -> RunResult:
        if runner_context.runner_id == "main":
            recent = runner_context.state_api.turn_list_recent("default", 5)
            task.payload["history"] = [turn.to_dict() for turn in recent]
        tools = runner_context.tool_api.tool_structured_list()
        stream_chunks: List[str] = []
        if runner_context.stream and hasattr(runner_context.llm_api, "collect_stream"):
            response, stream_chunks = runner_context.llm_api.collect_stream(task, tools)
        else:
            response = runner_context.llm_api.generate(task, tools)

        tool_results = []
        for call in response.tool_calls:
            tool_context = _build_tool_context(runner_context, task)
            try:
                result = runner_context.tool_api.tool_invoke(
                    call.tool_name, call.args, tool_context
                )
                tool_results.append({"tool": call.tool_name, "result": result})
            except ValueError as exc:
                tool_results.append(
                    {
                        "tool": call.tool_name,
                        "result": {
                            "error_type": "invalid_args",
                            "message": str(exc),
                        },
                    }
                )
            except Exception as exc:
                tool_results.append(
                    {
                        "tool": call.tool_name,
                        "result": {
                            "error_type": "tool_error",
                            "message": str(exc),
                        },
                    }
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
        messages = _history_to_messages(history)
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
        messages = _history_to_messages(history)
        tools = runner_context.tool_api.tool_structured_list()
        while step_context.budget.remaining_steps > 0:
            step_toolset = _discover_step_tools(
                task,
                step_context,
                "",
                "",
                runner_context,
                tools,
                messages,
                enforce_search=True,
            )
            logger.info(f"step_toolset: {step_toolset}")
            response, tool_results = _tool_loop_step(
                task,
                step_context,
                messages,
                runner_context,
                tools,
                step_toolset,
            )
            if response.user_output:
                step_context.last_result = response.user_output
                if response.user_output not in step_context.facts:
                    step_context.facts.append(response.user_output)
            step_context.budget.spent_steps += 1
            step_context.budget.remaining_steps -= 1
            logger.info(f"tool_results {tool_results}")
            if not tool_results:
                return RunResult(
                    user_output=response.user_output,
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
        payload={"message": prompt},
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
    tools: List[StructuredTool],
    step_toolset: Set[str],
) -> Tuple[LLMResponse, List[Dict[str, Any]]]:
    allowed_tools = [tool for tool in tools if tool.name in step_toolset]
    prompt = build_do_step_prompt(
        step_goal=step_goal,
        step_success_criteria=step_success_criteria,
        step_context=step_context.to_dict(),
        step_toolset=sorted(step_toolset),
    )
    messages: List[HumanMessage | AIMessage | ToolMessage] = [
        HumanMessage(content=prompt)
    ]
    do_task = Task(
        task_id=task.task_id,
        task_type="pdca.do",
        payload={"messages": messages},
        state="running",
    )
    response = runner_context.llm_api.generate(do_task, allowed_tools)
    messages.append(_ensure_ai_message(response))

    tool_results = []
    for call in response.tool_calls:
        if call.tool_name not in step_toolset:
            tool_results.append(
                {
                    "tool": call.tool_name,
                    "result": {"error_type": "tool_not_allowed"},
                    "tool_call_id": call.tool_call_id,
                }
            )
            continue
        tool_context = _build_tool_context(runner_context, task)
        try:
            result = runner_context.tool_api.tool_invoke(
                call.tool_name, call.args, tool_context
            )
            print("result:", result)
            tool_results.append(
                {
                    "tool": call.tool_name,
                    "result": result,
                    "tool_call_id": call.tool_call_id,
                }
            )
        except ValueError as exc:
            tool_results.append(
                {
                    "tool": call.tool_name,
                    "result": {
                        "error_type": "invalid_args",
                        "message": str(exc),
                    },
                    "tool_call_id": call.tool_call_id,
                }
            )
        except Exception as exc:
            tool_results.append(
                {
                    "tool": call.tool_name,
                    "result": {
                        "error_type": "tool_error",
                        "message": str(exc),
                    },
                    "tool_call_id": call.tool_call_id,
                }
            )

    if not response.tool_calls:
        return response, tool_results

    for tool_result in tool_results:
        messages.append(_tool_message_from_result(tool_result))
    followup_prompt = build_do_followup_prompt(
        step_goal=step_goal,
        step_success_criteria=step_success_criteria,
        step_context=step_context.to_dict(),
    )
    messages.append(HumanMessage(content=followup_prompt))
    followup_task = Task(
        task_id=task.task_id,
        task_type="pdca.do.followup",
        payload={"messages": messages},
        state="running",
    )
    followup_response = runner_context.llm_api.generate(followup_task, [])
    messages.append(_ensure_ai_message(followup_response))
    return followup_response, tool_results


def _tool_loop_step(
    task: Task,
    step_context: StepContext,
    messages: List[BaseMessage],
    runner_context: RunnerContext,
    tools: Sequence[StructuredTool],
    step_toolset: Set[str],
) -> Tuple[LLMResponse, List[Dict[str, Any]]]:
    payload = task.payload or {}
    user_message = payload.get("message") or payload.get("prompt") or ""
    allowed_tools = [tool for tool in tools if tool.name in step_toolset]
    prompt = build_tool_loop_prompt(
        user_message=user_message,
        step_context=step_context.to_dict(),
    )
    messages.append(HumanMessage(content=prompt))
    do_task = Task(
        task_id=task.task_id,
        task_type="tool_loop.step",
        payload={"messages": messages},
        state="running",
    )
    response = runner_context.llm_api.generate(do_task, allowed_tools)
    messages.append(_ensure_ai_message(response))
    tool_results = []
    for call in response.tool_calls:
        if call.tool_name not in step_toolset:
            tool_results.append(
                {
                    "tool": call.tool_name,
                    "result": {"error_type": "tool_not_allowed"},
                    "tool_call_id": call.tool_call_id,
                }
            )
            continue
        tool_context = _build_tool_context(runner_context, task)
        try:
            result = runner_context.tool_api.tool_invoke(
                call.tool_name, call.args, tool_context
            )
            tool_results.append(
                {
                    "tool": call.tool_name,
                    "result": result,
                    "tool_call_id": call.tool_call_id,
                }
            )
        except ValueError as exc:
            tool_results.append(
                {
                    "tool": call.tool_name,
                    "result": {
                        "error_type": "invalid_args",
                        "message": str(exc),
                    },
                    "tool_call_id": call.tool_call_id,
                }
            )
        except Exception as exc:
            tool_results.append(
                {
                    "tool": call.tool_name,
                    "result": {
                        "error_type": "tool_error",
                        "message": str(exc),
                    },
                    "tool_call_id": call.tool_call_id,
                }
            )
    if not response.tool_calls:
        return response, tool_results

    for tool_result in tool_results:
        messages.append(_tool_message_from_result(tool_result))
    followup_prompt = build_tool_loop_followup_prompt(
        user_message=user_message,
        step_context=step_context.to_dict(),
    )
    messages.append(HumanMessage(content=followup_prompt))
    followup_task = Task(
        task_id=task.task_id,
        task_type="tool_loop.followup",
        payload={"messages": messages},
        state="running",
    )
    followup_response = runner_context.llm_api.generate(followup_task, [])
    messages.append(_ensure_ai_message(followup_response))
    return followup_response, tool_results


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
        payload={"message": prompt},
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


def _history_to_messages(history: Sequence[Any]) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    for turn in history:
        if isinstance(turn, (HumanMessage, AIMessage)):
            messages.append(turn)
            continue
        if isinstance(turn, dict):
            user_message = turn.get("user_message")
            assistant_message = turn.get("assistant_message")
        else:
            user_message = getattr(turn, "user_message", None)
            assistant_message = getattr(turn, "assistant_message", None)
        if user_message:
            messages.append(HumanMessage(content=user_message))
        if assistant_message:
            messages.append(AIMessage(content=assistant_message))
    return messages


def _ensure_ai_message(response: LLMResponse) -> AIMessage:
    if isinstance(response.message, AIMessage):
        return response.message
    return AIMessage(content=response.user_output or "")


def _tool_message_from_result(tool_result: Dict[str, Any]) -> ToolMessage:
    tool_call_id = str(tool_result.get("tool_call_id") or tool_result.get("tool") or "")
    content = json.dumps(tool_result.get("result"), ensure_ascii=False)
    return ToolMessage(content=content, tool_call_id=tool_call_id)


def _safe_json_load(text: Optional[str]) -> Dict[str, Any]:
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
    tools: Sequence[StructuredTool],
    messages: Sequence[BaseMessage],
    enforce_search: bool = False,
) -> Set[str]:
    prompt = build_discover_tools_prompt(
        step_goal=step_goal,
        step_success_criteria=step_success_criteria,
        step_context=step_context.to_dict(),
        history=[{"role": msg.type, "content": msg.content} for msg in messages],
    )
    discover_task = Task(
        task_id=task.task_id,
        task_type="pdca.discover",
        payload={"message": prompt},
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
