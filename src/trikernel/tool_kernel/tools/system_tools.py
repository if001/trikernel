from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import InjectedState
from pydantic import Field
from typing_extensions import Annotated

from .prompts import build_step_goal_prompt
from ..runtime import ToolRuntime, get_runtime


def _require_runtime(state: dict) -> ToolRuntime:
    runtime_id = state.get("runtime_id") if isinstance(state, dict) else None
    if not isinstance(runtime_id, str) or not runtime_id:
        raise ValueError("runtime_id is required in state")
    runtime = get_runtime(runtime_id)
    if runtime is None:
        raise ValueError("runtime is required in tool runtime registry")
    return runtime


def step_goal(
    previous_goal: Annotated[
        Optional[str], Field(default=None, description="Previous goal.")
    ] = None,
    step_context: Annotated[
        Optional[Dict[str, object]],
        Field(default=None, description="Step context snapshot."),
    ] = None,
    user_message: Annotated[
        Optional[str], Field(default=None, description="User message.")
    ] = None,
    failure_reason: Annotated[
        Optional[str], Field(default=None, description="Failure reason.")
    ] = None,
    state: Annotated[dict, InjectedState] = {},
) -> Dict[str, Any]:
    runtime = _require_runtime(state)
    state_api = runtime.state_api
    task_id = state.get("task_id") if isinstance(state, dict) else None
    if not task_id:
        return {"step_goal": "", "error": "task_id_missing"}
    task = state_api.task_get(task_id)
    if not task:
        return {"step_goal": "", "error": "task_not_found"}
    payload_data = task.payload or {}
    fallback_goal = (
        payload_data.get("step_goal")
        or payload_data.get("user_message")
        or payload_data.get("message")
        or payload_data.get("prompt")
        or user_message
        or previous_goal
        or json.dumps(payload_data, ensure_ascii=False)
    )
    llm_api = runtime.tool_api.tool_llm_api()
    prompt = build_step_goal_prompt(
        previous_goal=previous_goal,
        failure_reason=failure_reason,
        step_context=step_context,
        user_message=user_message,
        task_payload=payload_data,
        history=[],
    )
    response_text = llm_api.generate(prompt, [])
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        return {"step_goal": fallback_goal, "rationale": response_text}
    return {
        "step_goal": parsed.get("step_goal") or fallback_goal,
        "rationale": parsed.get("rationale", ""),
    }


def build_system_tools() -> list[BaseTool]:
    return [
        StructuredTool.from_function(
            step_goal,
            name="step.goal",
            description=(
                "Propose/refine the next step goal given the current context (including failure reason).\n"
                "Use at the start of each tool-execution loop iteration to keep actions aligned with the user’s intent."
            ),
        )
    ]
