from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from trikernel.state_kernel.protocols import StateKernelAPI

from ..runtime import get_llm_api, get_state_api


def _require_state_api(state: dict) -> StateKernelAPI:
    state_api = state.get("state_api") if isinstance(state, dict) else None
    if state_api is None and isinstance(state, dict):
        runtime_id = state.get("runtime_id")
        if isinstance(runtime_id, str):
            state_api = get_state_api(runtime_id)
    if state_api is None:
        raise ValueError("state_api is required in state")
    return state_api


class UserRequestPayload(BaseModel):
    user_message: str = Field(..., description="User message for the main runner.")


class WorkPayload(BaseModel):
    message: str = Field(
        ..., description="Work instruction message(with details) for the worker."
    )
    run_at: Optional[str] = Field(
        default=None, description="ISO8601 timestamp for scheduling."
    )
    repeat_interval_seconds: Optional[int] = Field(
        default=None, description="Repeat interval in seconds (>= 3600)."
    )
    repeat_enabled: Optional[bool] = Field(
        default=None, description="Whether repeating work is enabled."
    )


class WorkRepeatPayload(BaseModel):
    message: str = Field(
        ..., description="Work instruction message(with details) for the worker."
    )
    repeat_interval_seconds: int = Field(
        ..., description="Repeat interval in seconds (>= 3600)."
    )
    repeat_enabled: Optional[bool] = Field(
        default=None, description="Whether repeating work is enabled."
    )


class WorkAtPayload(BaseModel):
    message: str = Field(
        ..., description="Work instruction message(with details) for the worker."
    )
    run_at: str = Field(..., description="ISO8601 timestamp for scheduling.")


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
    payload: WorkAtPayload


class TaskCreateWorkRepeatArgs(BaseModel):
    payload: WorkRepeatPayload


class TaskCreateNotificationArgs(BaseModel):
    payload: NotificationPayload


class TaskUpdateArgs(BaseModel):
    task_id: str = Field(..., description="Task id to update.")
    patch: Dict[str, object] = Field(..., description="Patch payload for the task.")


class TaskGetArgs(BaseModel):
    task_id: str = Field(..., description="Task id to fetch.")


class TaskListArgs(BaseModel):
    task_type: Optional[str] = Field(default=None, description="Filter by task type.")
    state: Optional[str] = Field(default="queued", description="Filter by state.")


class TaskClaimArgs(BaseModel):
    filter_by: Dict[str, object] = Field(..., description="Filter for claiming.")
    claimer_id: str = Field(..., description="Claimer id.")
    ttl_seconds: int = Field(..., description="Claim TTL in seconds.")


class TaskCompleteArgs(BaseModel):
    task_id: str = Field(..., description="Task id to complete.")


class TaskFailArgs(BaseModel):
    task_id: str = Field(..., description="Task id to fail.")
    error_info: Dict[str, object] = Field(..., description="Error payload.")


class ArtifactWriteArgs(BaseModel):
    media_type: str = Field(..., description="Artifact media type.")
    body: str = Field(..., description="Artifact body.")
    metadata: Dict[str, object] = Field(..., description="Artifact metadata.")


class ArtifactReadArgs(BaseModel):
    artifact_id: str = Field(..., description="Artifact id.")


class ArtifactExtractArgs(BaseModel):
    artifact_id: str = Field(..., description="Artifact id.")
    instructions: str = Field(..., description="Extraction instructions.")


class ArtifactSearchArgs(BaseModel):
    query: Dict[str, object] = Field(..., description="Metadata query.")


class EmptyArgs(BaseModel):
    pass


def task_create_user_request(
    payload: TaskCreateUserRequestArgs,
    state: Annotated[dict, InjectedState],
) -> str:
    state_api = _require_state_api(state)
    return state_api.task_create("user_request", payload.payload.model_dump())


def task_create_work(
    payload: TaskCreateWorkArgs,
    state: Annotated[dict, InjectedState],
) -> str:
    state_api = _require_state_api(state)
    return state_api.task_create("work", payload.payload.model_dump())


def task_create_work_at(
    payload: TaskCreateWorkAtArgs,
    state: Annotated[dict, InjectedState],
) -> str:
    run_at = payload.payload.run_at
    _validate_run_at(str(run_at))
    state_api = _require_state_api(state)
    return state_api.task_create("work", payload.payload.model_dump())


def task_create_work_repeat(
    payload: TaskCreateWorkRepeatArgs,
    state: Annotated[dict, InjectedState],
) -> str:
    interval = payload.payload.repeat_interval_seconds
    if int(interval) < 3600:
        raise ValueError("repeat_interval_seconds must be >= 3600")
    payload_dict = payload.payload.model_dump()
    payload_dict.setdefault("repeat_enabled", True)
    state_api = _require_state_api(state)
    return state_api.task_create("work", payload_dict)


def task_create_notification(
    payload: TaskCreateNotificationArgs,
    state: Annotated[dict, InjectedState],
) -> str:
    state_api = _require_state_api(state)
    return state_api.task_create("notification", payload.payload.model_dump())


def task_update(
    payload: TaskUpdateArgs,
    state: Annotated[dict, InjectedState],
) -> Optional[Dict[str, Any]]:
    state_api = _require_state_api(state)
    task = state_api.task_update(payload.task_id, payload.patch)
    return task.to_dict() if task else None


def task_get(
    payload: TaskGetArgs,
    state: Annotated[dict, InjectedState],
) -> Optional[Dict[str, Any]]:
    state_api = _require_state_api(state)
    task = state_api.task_get(payload.task_id)
    return task.to_dict() if task else None


def task_list(
    payload: TaskListArgs,
    state: Annotated[dict, InjectedState],
) -> List[Dict[str, Any]]:
    state_api = _require_state_api(state)
    return [
        task.to_dict() for task in state_api.task_list(payload.task_type, payload.state)
    ]


def task_claim(
    payload: TaskClaimArgs,
    state: Annotated[dict, InjectedState],
) -> Optional[str]:
    state_api = _require_state_api(state)
    return state_api.task_claim(
        payload.filter_by, payload.claimer_id, payload.ttl_seconds
    )


def task_complete(
    payload: TaskCompleteArgs,
    state: Annotated[dict, InjectedState],
) -> Optional[Dict[str, Any]]:
    state_api = _require_state_api(state)
    task = state_api.task_complete(payload.task_id)
    return task.to_dict() if task else None


def task_fail(
    payload: TaskFailArgs,
    state: Annotated[dict, InjectedState],
) -> Optional[Dict[str, Any]]:
    state_api = _require_state_api(state)
    task = state_api.task_fail(payload.task_id, payload.error_info)
    return task.to_dict() if task else None


def artifact_write(
    payload: ArtifactWriteArgs,
    state: Annotated[dict, InjectedState],
) -> Dict[str, Any]:
    state_api = _require_state_api(state)
    artifact_id = state_api.artifact_write(
        payload.media_type, payload.body, payload.metadata
    )


    return {"artifact_id": artifact_id, "content_path": }


def artifact_read(
    payload: ArtifactReadArgs,
    state: Annotated[dict, InjectedState],
) -> Optional[Dict[str, Any]]:
    state_api = _require_state_api(state)
    artifact = state_api.artifact_read(payload.artifact_id)
    return artifact.to_full_dict() if artifact else None


def artifact_extract(
    payload: ArtifactExtractArgs,
    state: Annotated[dict, InjectedState],
) -> Dict[str, Any]:
    state_api = _require_state_api(state)
    artifact = state_api.artifact_read(payload.artifact_id)
    if not artifact:
        return {"artifact_id": payload.artifact_id, "error": "not_found"}

    llm_api = state.get("llm_api") if isinstance(state, dict) else None
    if llm_api is None and isinstance(state, dict):
        runtime_id = state.get("runtime_id")
        if isinstance(runtime_id, str):
            llm_api = get_llm_api(runtime_id)
    if not llm_api:
        return {"artifact_id": payload.artifact_id, "error": "llm_api_required"}

    prompt = f"extract instructions: {payload.instructions}\n{artifact.body}"
    extracted = llm_api.generate(prompt, [])
    return {"artifact_id": payload.artifact_id, "result": extracted}


def artifact_search(
    payload: ArtifactSearchArgs,
    state: Annotated[dict, InjectedState],
) -> List[Dict[str, Any]]:
    state_api = _require_state_api(state)
    return [
        artifact.to_small_dict()
        for artifact in state_api.artifact_search(payload.query)
    ]


def artifact_list(
    payload: EmptyArgs,
    state: Annotated[dict, InjectedState],
) -> List[Dict[str, Any]]:
    _ = payload
    state_api = _require_state_api(state)
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


def build_state_tools() -> List[BaseTool]:
    return [
        # StructuredTool.from_function(
        #     task_create_user_request,
        #     name="task.create_user_request",
        #     description="Create a user_request task.",
        # ),
        StructuredTool.from_function(
            task_create_work,
            name="task.create_work",
            description=(
                "Start a worker deep-work job for investigations that may exceed the main agent’s per-step tool-call budget.\n"
                "Use when the main loop needs to offload long-running research / multi-hop browsing / heavy extraction beyond allowed tool iterations.\n"
                "Not for scheduling; for time-based or periodic runs use task.create_work_at / task.create_work_repeat.\n"
                "The worker runtime will always emit a notification at the end (not via this tool call)."
            ),
        ),
        StructuredTool.from_function(
            task_create_work_at,
            name="task.create_work_at",
            description=(
                "Schedule a worker job at a specific run_at (ISO8601).\n"
                "Use for reminders, delayed checks, or actions that must happen at a certain time.\n"
                "Not for deep-work offloading; use task.create_work if the goal is to exceed main-loop tool budget."
            ),
        ),
        StructuredTool.from_function(
            task_create_work_repeat,
            name="task.create_work_repeat",
            description=(
                "Schedule a repeating worker job at a fixed interval (repeat_interval_seconds >= 3600).\n"
                "Use for periodic monitoring/digests/maintenance.\n"
                "Each run will end with a notification emitted by the worker runtime."
            ),
        ),
        # StructuredTool.from_function(
        #     task_create_notification,
        #     name="task.create_notification",
        #     description="Create a notification task.",
        # ),
        # StructuredTool.from_function(
        #     task_update,
        #     name="task.update",
        #     description="Update a task with a patch payload.",
        # ),
        StructuredTool.from_function(
            task_get,
            name="task.get",
            description="Fetch a task by id for debugging, tracing, or runner logic.",
        ),
        StructuredTool.from_function(
            task_list,
            name="task.list",
            description="List tasks by type/state for runner polling, dashboards, or maintenance.",
        ),
        # StructuredTool.from_function(
        #     task_claim,
        #     name="task.claim",
        #     description="Claim a task for execution.",
        # ),
        # StructuredTool.from_function(
        #     task_complete,
        #     name="task.complete",
        #     description="Mark a task as complete.",
        # ),
        # StructuredTool.from_function(
        #     task_fail,
        #     name="task.fail",
        #     description="Mark a task as failed.",
        # ),
        # StructuredTool.from_function(
        #     artifact_write,
        #     name="artifact.write",
        #     description=(
        #         "Persist an artifact (text body + media_type + metadata) for later retrieval via semantic search and reuse across steps/workers.\n"
        #         "Use to store fetched web content, intermediate results, or user-requested temporary notes."
        #     ),
        # ),
        # StructuredTool.from_function(
        #     artifact_read,
        #     name="artifact.read",
        #     description=(
        #         "Read a stored artifact by id (returns full body).\n"
        #         "Use when you already know the exact artifact to reuse."
        #     ),
        # ),
        # StructuredTool.from_function(
        #     artifact_extract,
        #     name="artifact.extract",
        #     description=(
        #         "Run LLM-based extraction over an artifact specified by id using provided instructions (e.g., pull facts, make bullet notes, extract entities).\n"
        #         "Use after artifact.search when the body is long and you only need specific information; prefer this over artifact.read when possible."
        #     ),
        # ),
        # StructuredTool.from_function(
        #     artifact_search,
        #     name="artifact.search",
        #     description=(
        #         "Semantic search over stored artifacts using an embedding query against artifact bodies.\n"
        #         "Returns artifact IDs only (and optionally scores if available).\n"
        #         "Use to locate relevant artifacts, then call artifact.read to fetch the body or artifact.extract to pull targeted information."
        #     ),
        # ),
        # StructuredTool.from_function(
        #     artifact_list,
        #     name="artifact.list",
        #     description="List stored artifacts for inspection/debugging. Prefer artifact.search for finding relevant artifacts by meaning.",
        # ),
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
