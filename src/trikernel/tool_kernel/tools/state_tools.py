from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from ..models import ToolContext
from .structured_tools import build_structured_tool


def _require_state_api(context: ToolContext) -> Any:
    if context is None or context.state_api is None:
        raise ValueError("state_api is required in ToolContext")
    return context.state_api


class UserRequestPayload(BaseModel):
    user_message: str = Field(..., description="User message for the main runner.")


class WorkPayload(BaseModel):
    message: str = Field(..., description="Work instruction message for the worker.")
    run_at: Optional[str] = Field(
        default=None, description="ISO8601 timestamp for scheduling."
    )
    repeat_interval_seconds: Optional[int] = Field(
        default=None, description="Repeat interval in seconds (>= 3600)."
    )
    repeat_enabled: Optional[bool] = Field(
        default=None, description="Whether repeating work is enabled."
    )


class NotificationPayload(BaseModel):
    message: str = Field(..., description="Notification message to deliver.")
    severity: Optional[str] = Field(default=None, description="Optional severity.")
    related_task_id: Optional[str] = Field(
        default=None, description="Related task id, if any."
    )


class TaskCreateUserRequestArgs(BaseModel):
    payload: UserRequestPayload


class TaskCreateWorkArgs(BaseModel):
    payload: WorkPayload


class TaskCreateWorkAtArgs(BaseModel):
    payload: WorkPayload


class TaskCreateWorkRepeatArgs(BaseModel):
    payload: WorkPayload


class TaskCreateNotificationArgs(BaseModel):
    payload: NotificationPayload


class TaskUpdateArgs(BaseModel):
    task_id: str
    patch: Dict[str, object]


class TaskGetArgs(BaseModel):
    task_id: str


class TaskListArgs(BaseModel):
    task_type: Optional[str] = None
    state: Optional[str] = "queued"


class TaskClaimArgs(BaseModel):
    filter_by: Dict[str, object]
    claimer_id: str
    ttl_seconds: int


class TaskCompleteArgs(BaseModel):
    task_id: str


class TaskFailArgs(BaseModel):
    task_id: str
    error_info: Dict[str, object]


class ArtifactWriteArgs(BaseModel):
    media_type: str
    body: str
    metadata: Dict[str, object]


class ArtifactReadArgs(BaseModel):
    artifact_id: str


class ArtifactExtractArgs(BaseModel):
    artifact_id: str
    instructions: str


class ArtifactSearchArgs(BaseModel):
    query: Dict[str, object]


class EmptyArgs(BaseModel):
    pass


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


def build_state_tools() -> List[tuple[BaseTool, Any]]:
    return [
        (
            build_structured_tool(
                task_create_user_request,
                name="task.create_user_request",
                description="Create a user_request task.",
                args_schema=TaskCreateUserRequestArgs,
            ),
            task_create_user_request,
        ),
        (
            build_structured_tool(
                task_create_work,
                name="task.create_work",
                description="Create a work task.",
                args_schema=TaskCreateWorkArgs,
            ),
            task_create_work,
        ),
        (
            build_structured_tool(
                task_create_work_at,
                name="task.create_work_at",
                description="Create a scheduled work task.",
                args_schema=TaskCreateWorkAtArgs,
            ),
            task_create_work_at,
        ),
        (
            build_structured_tool(
                task_create_work_repeat,
                name="task.create_work_repeat",
                description="Create a repeating work task.",
                args_schema=TaskCreateWorkRepeatArgs,
            ),
            task_create_work_repeat,
        ),
        (
            build_structured_tool(
                task_create_notification,
                name="task.create_notification",
                description="Create a notification task.",
                args_schema=TaskCreateNotificationArgs,
            ),
            task_create_notification,
        ),
        (
            build_structured_tool(
                task_update,
                name="task.update",
                description="Update a task with a patch payload.",
                args_schema=TaskUpdateArgs,
            ),
            task_update,
        ),
        (
            build_structured_tool(
                task_get,
                name="task.get",
                description="Get a task by id.",
                args_schema=TaskGetArgs,
            ),
            task_get,
        ),
        (
            build_structured_tool(
                task_list,
                name="task.list",
                description="List tasks by type/state.",
                args_schema=TaskListArgs,
            ),
            task_list,
        ),
        (
            build_structured_tool(
                task_claim,
                name="task.claim",
                description="Claim a task for execution.",
                args_schema=TaskClaimArgs,
            ),
            task_claim,
        ),
        (
            build_structured_tool(
                task_complete,
                name="task.complete",
                description="Mark a task as complete.",
                args_schema=TaskCompleteArgs,
            ),
            task_complete,
        ),
        (
            build_structured_tool(
                task_fail,
                name="task.fail",
                description="Mark a task as failed.",
                args_schema=TaskFailArgs,
            ),
            task_fail,
        ),
        (
            build_structured_tool(
                artifact_write,
                name="artifact.write",
                description="Write an artifact and return its id.",
                args_schema=ArtifactWriteArgs,
            ),
            artifact_write,
        ),
        (
            build_structured_tool(
                artifact_read,
                name="artifact.read",
                description="Read an artifact by id.",
                args_schema=ArtifactReadArgs,
            ),
            artifact_read,
        ),
        (
            build_structured_tool(
                artifact_extract,
                name="artifact.extract",
                description="Extract information from an artifact using the LLM.",
                args_schema=ArtifactExtractArgs,
            ),
            artifact_extract,
        ),
        (
            build_structured_tool(
                artifact_search,
                name="artifact.search",
                description="Search artifacts by metadata.",
                args_schema=ArtifactSearchArgs,
            ),
            artifact_search,
        ),
        (
            build_structured_tool(
                artifact_list,
                name="artifact.list",
                description="List stored artifacts.",
                args_schema=EmptyArgs,
            ),
            artifact_list,
        ),
    ]


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
