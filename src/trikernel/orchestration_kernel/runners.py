from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import Budget, LLMResponse, RunResult, RunnerContext, StepContext
from .protocols import Runner
from ..state_kernel.models import Task, utc_now
from ..tool_kernel.models import ToolContext
from .logging import get_logger

logger = get_logger(__name__)


def _tool_descriptions(tool_api: Any) -> List[Dict[str, Any]]:
    return [
        {
            "tool_name": tool.tool_name,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "output_schema": tool.output_schema,
            "effects": tool.effects,
        }
        for tool in tool_api.tool_list()
    ]


class SingleTurnRunner(Runner):
    def run(self, task: Task, runner_context: RunnerContext) -> RunResult:
        if runner_context.runner_id == "main":
            recent = runner_context.state_api.turn_list_recent("default", 5)
            task.payload["history"] = [turn.to_dict() for turn in recent]
        tools = _tool_descriptions(runner_context.tool_api)
        stream_chunks: List[str] = []
        if runner_context.stream and hasattr(runner_context.llm_api, "collect_stream"):
            response, stream_chunks = runner_context.llm_api.collect_stream(task, tools)
        else:
            response = runner_context.llm_api.generate(task, tools)

        tool_results = []
        for call in response.tool_calls:
            tool_context = ToolContext(
                runner_id=runner_context.runner_id,
                task_id=task.task_id,
                state_api=runner_context.state_api,
                now=utc_now(),
            )
            result = runner_context.tool_api.tool_invoke(
                call.tool_name, call.args, tool_context
            )
            tool_results.append({"tool": call.tool_name, "result": result})

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
        tools = _tool_descriptions(runner_context.tool_api)
        for v in tools:
            print(v["tool_name"])
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
                            "artifact_refs": list(step_context.artifact_refs),
                        },
                    )
                    final_message = None
                return RunResult(
                    user_output=final_message,
                    task_state="done",
                    artifact_refs=list(step_context.artifact_refs),
                    error=None,
                    stream_chunks=[],
                )

        return RunResult(
            user_output=None,
            task_state="failed",
            artifact_refs=list(step_context.artifact_refs),
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
        artifact_refs=list(context_payload.get("artifact_refs", [])),
        budget=budget,
    )


def _plan_step(
    task: Task, step_context: StepContext, runner_context: RunnerContext
) -> Tuple[str, str]:
    history = runner_context.state_api.turn_list_recent("default", 5)
    prompt = (
        "You are planning the next step for a task.\n"
        "Create a short, concrete step_goal and objective success criteria.\n"
        "Respond in JSON with keys `step_goal` and `step_success_criteria`.\n"
        "Guidance:\n"
        "- step_goal: one clear action to take next.\n"
        "- step_success_criteria: measurable or verifiable outcome.\n"
        f"Task payload: {json.dumps(task.payload, ensure_ascii=False)}\n"
        f"Step context: {json.dumps(step_context.to_dict(), ensure_ascii=False)}\n"
        f"Recent turns: {json.dumps([t.to_dict() for t in history], ensure_ascii=False)}"
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
    tools: List[Dict[str, Any]],
    step_toolset: Set[str],
) -> Tuple[LLMResponse, List[Dict[str, Any]]]:
    allowed_tools = [tool for tool in tools if tool["tool_name"] in step_toolset]
    prompt = (
        "You are executing the next step for a task.\n"
        "Decide whether to call a tool. If needed, call exactly the tools required. "
        "If no tools are required, respond to the user directly.\n"
        "Responses to users must be in Japanese. Do not output internal terminology as-is.\n"
        "Only call tools from the allowed list.\n"
        f"Step goal: {step_goal}\n"
        f"Success criteria: {step_success_criteria}\n"
        f"Step context: {json.dumps(step_context.to_dict(), ensure_ascii=False)}\n"
        f"Allowed tools: {sorted(step_toolset)}"
    )
    do_task = Task(
        task_id=task.task_id,
        task_type="pdca.do",
        payload={"message": prompt},
        state="running",
    )
    print("do_step", prompt)
    response = runner_context.llm_api.generate(do_task, allowed_tools)

    tool_results = []
    for call in response.tool_calls:
        if call.tool_name not in step_toolset:
            tool_results.append(
                {
                    "tool": call.tool_name,
                    "result": {"error": "tool_not_allowed"},
                }
            )
            continue
        tool_context = ToolContext(
            runner_id=runner_context.runner_id,
            task_id=task.task_id,
            state_api=runner_context.state_api,
            now=utc_now(),
        )
        result = runner_context.tool_api.tool_invoke(
            call.tool_name, call.args, tool_context
        )
        tool_results.append({"tool": call.tool_name, "result": result})

    if not response.tool_calls:
        return response, tool_results

    followup_prompt = (
        "Tool execution finished. Summarize results and respond to the user.\n"
        "Responses to users must be in Japanese. Do not output internal terminology as-is.\n"
        f"Step goal: {step_goal}\n"
        f"Success criteria: {step_success_criteria}\n"
        f"Step context: {json.dumps(step_context.to_dict(), ensure_ascii=False)}\n"
        f"Tool results: {json.dumps(tool_results, ensure_ascii=False)}"
    )
    followup_task = Task(
        task_id=task.task_id,
        task_type="pdca.do.followup",
        payload={"message": followup_prompt},
        state="running",
    )
    followup_response = runner_context.llm_api.generate(followup_task, [])
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
    prompt = (
        "Evaluate whether the step_goal was achieved.\n"
        "Respond in JSON with keys `achieved` (boolean), `evaluation` (string), "
        "and `gaps` (array of strings).\n"
        "Use the step_success_criteria as the basis for judgment.\n"
        "If the do output is too short or insufficient to satisfy the criteria, "
        "set achieved=false and include the gap.\n"
        f"Step goal: {step_goal}\n"
        f"Success criteria: {step_success_criteria}\n"
        f"Step context: {json.dumps(step_context.to_dict(), ensure_ascii=False)}\n"
        f"Do output: {user_output}\n"
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
    tools: List[Dict[str, Any]],
) -> Set[str]:
    history = runner_context.state_api.turn_list_recent("default", 5)
    prompt = (
        "Select tools that are necessary to complete the step_goal.\n"
        "First propose search queries to find relevant tools, then select tool names.\n"
        "Return JSON with keys `search_queries` (array of strings) and "
        "`selected_tools` (array of tool names).\n"
        f"Step goal: {step_goal}\n"
        f"Success criteria: {step_success_criteria}\n"
        f"Step context: {json.dumps(step_context.to_dict(), ensure_ascii=False)}\n"
        f"Recent turns: {json.dumps([t.to_dict() for t in history], ensure_ascii=False)}"
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
