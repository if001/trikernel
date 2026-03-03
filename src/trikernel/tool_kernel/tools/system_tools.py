from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from ..prompts import build_step_goal_prompt
from ..runtime import get_llm_api, get_state_api


def _require_state_api(state: dict) -> Any:
    state_api = state.get("state_api") if isinstance(state, dict) else None
    if state_api is None and isinstance(state, dict):
        runtime_id = state.get("runtime_id")
        if isinstance(runtime_id, str):
            state_api = get_state_api(runtime_id)
    if state_api is None:
        raise ValueError("state_api is required in state")
    return state_api


def _require_llm_api(state: dict) -> Any:
    llm_api = state.get("llm_api") if isinstance(state, dict) else None
    if llm_api is None and isinstance(state, dict):
        runtime_id = state.get("runtime_id")
        if isinstance(runtime_id, str):
            llm_api = get_llm_api(runtime_id)
    if llm_api is None:
        raise ValueError("llm_api is required in state")
    return llm_api


class StepGoalArgs(BaseModel):
    previous_goal: Optional[str] = Field(default=None, description="Previous goal.")
    step_context: Optional[Dict[str, object]] = Field(
        default=None, description="Step context snapshot."
    )
    user_message: Optional[str] = Field(default=None, description="User message.")
    failure_reason: Optional[str] = Field(default=None, description="Failure reason.")


def step_goal(
    previous_goal: Optional[str] = None,
    step_context: Optional[Dict[str, object]] = None,
    user_message: Optional[str] = None,
    failure_reason: Optional[str] = None,
    *,
    state: Annotated[dict, InjectedState],
) -> Dict[str, Any]:
    state_api = _require_state_api(state)
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
    llm_api = _require_llm_api(state)
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
            args_schema=StepGoalArgs,
        )
    ]
