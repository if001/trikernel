from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def build_step_goal_prompt(
    previous_goal: Optional[str],
    failure_reason: Optional[str],
    step_context: Optional[Dict[str, Any]],
    user_message: Optional[str],
    task_payload: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> str:
    return (
        "You are deciding the current step goal for a task.\n"
        "If a previous_goal exists, keep it unless there is a clear reason to adjust it.\n"
        "If the previous step failed, incorporate the failure_reason or open issues to refine the goal.\n"
        "Use the latest user_message or worker input when it is more specific than previous_goal.\n"
        "Respond in JSON with keys `step_goal` and `rationale`.\n"
        f"Previous goal: {previous_goal}\n"
        f"Failure reason: {failure_reason}\n"
        f"Step context: {json.dumps(step_context or {}, ensure_ascii=False)}\n"
        f"User/worker input: {user_message}\n"
        f"Task payload: {json.dumps(task_payload, ensure_ascii=False)}\n"
        f"Recent turns: {json.dumps(history, ensure_ascii=False)}"
    )
