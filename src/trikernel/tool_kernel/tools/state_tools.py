from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..models import ToolContext


def _require_state_api(context: ToolContext) -> Any:
    if context is None or context.state_api is None:
        raise ValueError("state_api is required in ToolContext")
    return context.state_api


def task_create_user_request(payload: Dict[str, Any], *, context: ToolContext) -> str:
    state_api = _require_state_api(context)
    return state_api.task_create("user_request", payload)


def task_create_work(payload: Dict[str, Any], *, context: ToolContext) -> str:
    state_api = _require_state_api(context)
    return state_api.task_create("work", payload)


def task_create_work_at(payload: Dict[str, Any], *, context: ToolContext) -> str:
    run_at = payload.get("run_at")
    if not run_at:
        raise ValueError("run_at is required")
    _validate_run_at(str(run_at))
    state_api = _require_state_api(context)
    return state_api.task_create("work", payload)


def task_create_work_repeat(payload: Dict[str, Any], *, context: ToolContext) -> str:
    interval = payload.get("repeat_interval_seconds")
    if interval is None:
        raise ValueError("repeat_interval_seconds is required")
    if int(interval) < 3600:
        raise ValueError("repeat_interval_seconds must be >= 3600")
    payload = dict(payload)
    payload.setdefault("repeat_enabled", True)
    state_api = _require_state_api(context)
    return state_api.task_create("work", payload)


def task_create_notification(payload: Dict[str, Any], *, context: ToolContext) -> str:
    state_api = _require_state_api(context)
    return state_api.task_create("notification", payload)


def task_update(
    task_id: str, patch: Dict[str, Any], *, context: ToolContext
) -> Optional[Dict[str, Any]]:
    state_api = _require_state_api(context)
    task = state_api.task_update(task_id, patch)
    return task.to_dict() if task else None


def task_get(task_id: str, *, context: ToolContext) -> Optional[Dict[str, Any]]:
    state_api = _require_state_api(context)
    task = state_api.task_get(task_id)
    return task.to_dict() if task else None


def task_list(
    task_type: Optional[str] = None,
    state: Optional[str] = "queued",
    *,
    context: ToolContext,
) -> List[Dict[str, Any]]:
    state_api = _require_state_api(context)
    return [task.to_dict() for task in state_api.task_list(task_type, state)]


def task_claim(
    filter_by: Dict[str, Any],
    claimer_id: str,
    ttl_seconds: int,
    *,
    context: ToolContext,
) -> Optional[str]:
    state_api = _require_state_api(context)
    return state_api.task_claim(filter_by, claimer_id, ttl_seconds)


def task_complete(
    task_id: str, *, context: ToolContext
) -> Optional[Dict[str, Any]]:
    state_api = _require_state_api(context)
    task = state_api.task_complete(task_id)
    return task.to_dict() if task else None


def task_fail(
    task_id: str, error_info: Dict[str, Any], *, context: ToolContext
) -> Optional[Dict[str, Any]]:
    state_api = _require_state_api(context)
    task = state_api.task_fail(task_id, error_info)
    return task.to_dict() if task else None


def artifact_write(
    media_type: str, body: str, metadata: Dict[str, Any], *, context: ToolContext
) -> str:
    state_api = _require_state_api(context)
    return state_api.artifact_write(media_type, body, metadata)


def artifact_read(
    artifact_id: str, *, context: ToolContext
) -> Optional[Dict[str, Any]]:
    state_api = _require_state_api(context)
    artifact = state_api.artifact_read(artifact_id)
    return artifact.to_dict() if artifact else None


def artifact_extract(
    artifact_id: str,
    instructions: str,
    *,
    context: ToolContext,
) -> Dict[str, Any]:
    state_api = _require_state_api(context)
    artifact = state_api.artifact_read(artifact_id)
    if not artifact:
        return {"error": "artifact_not_found"}
    if not context.llm_api:
        return {"error": "llm_api_required"}
    prompt = (
        "Extract the requested information from the artifact content.\n"
        f"Instructions: {instructions}\n"
        f"Artifact content: {artifact.body}\n"
        "Return only the extracted result."
    )
    extracted = context.llm_api.generate(prompt, [])
    return {"artifact_id": artifact_id, "result": extracted}


def artifact_search(
    query: Dict[str, Any], *, context: ToolContext
) -> List[Dict[str, Any]]:
    state_api = _require_state_api(context)
    return [artifact.to_dict() for artifact in state_api.artifact_search(query)]


def artifact_list(*, context: ToolContext) -> List[Dict[str, Any]]:
    state_api = _require_state_api(context)
    artifacts = state_api.artifact_list()
    result = []
    for artifact in artifacts:
        result.append(
            {
                "artifact_id": artifact.artifact_id,
                "metadata": dict(artifact.metadata),
                "created_at": artifact.created_at,
                "body": artifact.body[:100],
            }
        )
    return result


def turn_list_recent(
    conversation_id: str, limit: int, *, context: ToolContext
) -> List[Dict[str, Any]]:
    state_api = _require_state_api(context)
    return [turn.to_dict() for turn in state_api.turn_list_recent(conversation_id, limit)]


def state_tool_functions() -> Dict[str, Any]:
    return {
        "task.create_user_request": task_create_user_request,
        "task.create_work": task_create_work,
        "task.create_work_at": task_create_work_at,
        "task.create_work_repeat": task_create_work_repeat,
        "task.create_notification": task_create_notification,
        "task.update": task_update,
        "task.get": task_get,
        "task.list": task_list,
        "task.claim": task_claim,
        "task.complete": task_complete,
        "task.fail": task_fail,
        "artifact.write": artifact_write,
        "artifact.read": artifact_read,
        "artifact.extract": artifact_extract,
        "artifact.search": artifact_search,
        "artifact.list": artifact_list,
        "turn.list_recent": turn_list_recent,
    }


def _validate_run_at(run_at: str) -> None:
    try:
        parsed = datetime.fromisoformat(run_at)
    except ValueError as exc:
        raise ValueError("run_at must be ISO8601 format") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if parsed < now:
        raise ValueError("run_at must be in the future")
    if parsed > now + timedelta(days=365):
        raise ValueError("run_at must be within 1 year")
