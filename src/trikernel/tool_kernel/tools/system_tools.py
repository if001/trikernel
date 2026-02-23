from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from ..models import ToolContext
from ..prompts import build_step_goal_prompt
from .structured_tools import build_structured_tool


def _require_state_api(context: ToolContext) -> Any:
    if context is None or context.state_api is None:
        raise ValueError("state_api is required in ToolContext")
    return context.state_api


class StepGoalArgs(BaseModel):
    previous_goal: Optional[str] = None
    step_context: Optional[Dict[str, object]] = None
    user_message: Optional[str] = None
    failure_reason: Optional[str] = None


def step_goal(
    previous_goal: Optional[str] = None,
    step_context: Optional[Dict[str, Any]] = None,
    user_message: Optional[str] = None,
    failure_reason: Optional[str] = None,
    *,
    context: ToolContext,
) -> Dict[str, Any]:
    state_api = _require_state_api(context)
    if not context.task_id:
        return {"step_goal": "", "error": "task_id_missing"}
    task = state_api.task_get(context.task_id)
    if not task:
        return {"step_goal": "", "error": "task_not_found"}
    payload = task.payload or {}
    fallback_goal = (
        payload.get("step_goal")
        or payload.get("user_message")
        or payload.get("message")
        or payload.get("prompt")
        or user_message
        or previous_goal
        or json.dumps(payload, ensure_ascii=False)
    )
    llm_api = context.llm_api
    if not llm_api:
        return {"step_goal": fallback_goal}
    prompt = build_step_goal_prompt(
        previous_goal=previous_goal,
        failure_reason=failure_reason,
        step_context=step_context,
        user_message=user_message,
        task_payload=payload,
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


def build_system_tools() -> list[tuple[BaseTool, Any]]:
    return [
        (
            build_structured_tool(
                step_goal,
                name="step.goal",
                description="Generate or refine the next step goal.",
                args_schema=StepGoalArgs,
            ),
            step_goal,
        )
    ]
