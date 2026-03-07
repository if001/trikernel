from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import InjectedState
from pydantic import Field
from typing_extensions import Annotated

from trikernel.utils.logging import get_logger
from trikernel.utils.time_utils import validate_run_at_future

from ._shared import require_state_api, require_tool_llm

logger = get_logger(__name__)


def task_create_user_request(
    user_message: Annotated[
        str, Field(..., description="User message for the main runner.")
    ],
    state: Annotated[dict, InjectedState] = {},
) -> str:
    state_api = require_state_api(state)
    return state_api.task_create("user_request", {"user_message": user_message})


def task_create_work(
    message: Annotated[
        str,
        Field(
            ..., description="Work instruction message(with details) for the worker."
        ),
    ],
    state: Annotated[dict, InjectedState] = {},
) -> str:
    _ensure_not_worker(state)
    state_api = require_state_api(state)
    payload = {
        "message": message,
    }
    logger.info("!!!!!!!!!!!!!! create new task !!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger.info(f"message")
    return state_api.task_create("work", payload)


def task_create_work_at(
    message: Annotated[
        str,
        Field(
            ..., description="Work instruction message(with details) for the worker."
        ),
    ],
    run_at: Annotated[str, Field(..., description="ISO8601 timestamp for scheduling.")],
    state: Annotated[dict, InjectedState] = {},
) -> str:
    _validate_run_at(str(run_at))
    _ensure_not_worker(state)
    state_api = require_state_api(state)
    logger.info("!!!!!!!!!!!!!! create new task !!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger.info(f"message")
    return state_api.task_create("work", {"message": message, "run_at": run_at})


def task_create_work_repeat(
    message: Annotated[
        str,
        Field(
            ..., description="Work instruction message(with details) for the worker."
        ),
    ],
    repeat_interval_seconds: Annotated[
        int, Field(..., description="Repeat interval in seconds (>= 3600).")
    ],
    repeat_enabled: Annotated[
        Optional[bool],
        Field(default=None, description="Whether repeating work is enabled."),
    ] = None,
    state: Annotated[dict, InjectedState] = {},
) -> str:
    _ensure_not_worker(state)
    if repeat_enabled is not None:
        logger.error(
            "create_work_repeat does not accept repeat_enabled=%s", repeat_enabled
        )
        raise ValueError("create_work_repeat does not accept repeat_enabled")
    if int(repeat_interval_seconds) < 3600:
        raise ValueError("repeat_interval_seconds must be >= 3600")
    payload = {
        "message": message,
        "repeat_interval_seconds": repeat_interval_seconds,
        "repeat_enabled": True,
    }
    state_api = require_state_api(state)
    logger.info("!!!!!!!!!!!!!! create new task !!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger.info(f"message")
    return state_api.task_create("work", payload)


def task_create_notification(
    message: Annotated[str, Field(..., description="Notification message to deliver.")],
    severity: Annotated[
        Optional[str], Field(default=None, description="Optional severity.")
    ] = None,
    related_task_id: Annotated[
        Optional[str], Field(default=None, description="Related task id, if any.")
    ] = None,
    state: Annotated[dict, InjectedState] = {},
) -> str:
    state_api = require_state_api(state)
    payload = {
        "message": message,
        "severity": severity,
        "related_task_id": related_task_id,
    }
    return state_api.task_create("notification", payload)


def task_update(
    task_id: Annotated[str, Field(..., description="Task id to update.")],
    patch: Annotated[
        Dict[str, object], Field(..., description="Patch payload for the task.")
    ],
    state: Annotated[dict, InjectedState] = {},
) -> Optional[Dict[str, Any]]:
    _validate_run_at_patch(patch)
    state_api = require_state_api(state)
    task = state_api.task_update(task_id, patch)
    return task.to_dict() if task else None


def _validate_run_at_patch(patch: Dict[str, object]) -> None:
    payload = patch.get("payload")
    if not isinstance(payload, dict):
        return
    run_at = payload.get("run_at")
    if run_at is None:
        return
    try:
        validate_run_at_future(str(run_at))
    except ValueError as exc:
        logger.error("invalid run_at in patch: %s", run_at)
        raise exc


def _ensure_not_worker(state: dict) -> None:
    task_type = _extract_task_type(state)
    if task_type == "work":
        logger.error("workers cannot create worker tasks")
        raise ValueError("workers cannot create worker tasks")


def _extract_task_type(state: dict) -> Optional[str]:
    task_type = state.get("task_type")
    if isinstance(task_type, str):
        return task_type
    step_context = state.get("step_context")
    if step_context is not None:
        value = getattr(step_context, "task_type", None)
        if isinstance(value, str):
            return value
    return None


def task_get(
    task_id: Annotated[str, Field(..., description="Task id to fetch.")],
    state: Annotated[dict, InjectedState] = {},
) -> Optional[Dict[str, Any]]:
    state_api = require_state_api(state)
    task = state_api.task_get(task_id)
    return task.to_dict() if task else None


def task_list(
    task_type: Annotated[
        Optional[str], Field(default=None, description="Filter by task type.")
    ] = None,
    task_state: Annotated[
        Optional[str], Field(default="queued", description="Filter by state.")
    ] = "queued",
    state: Annotated[dict, InjectedState] = {},
) -> List[Dict[str, Any]]:
    state_api = require_state_api(state)
    task_state = task_state if task_state else "queued"
    return [task.to_dict() for task in state_api.task_list(task_type, task_state)]


def task_claim(
    filter_by: Annotated[
        Dict[str, object], Field(..., description="Filter for claiming.")
    ],
    claimer_id: Annotated[str, Field(..., description="Claimer id.")],
    ttl_seconds: Annotated[int, Field(..., description="Claim TTL in seconds.")],
    state: Annotated[dict, InjectedState] = {},
) -> Optional[str]:
    state_api = require_state_api(state)
    return state_api.task_claim(filter_by, claimer_id, ttl_seconds)


def task_complete(
    task_id: Annotated[str, Field(..., description="Task id to complete.")],
    state: Annotated[dict, InjectedState] = {},
) -> Optional[Dict[str, Any]]:
    state_api = require_state_api(state)
    task = state_api.task_complete(task_id)
    return task.to_dict() if task else None


def task_fail(
    task_id: Annotated[str, Field(..., description="Task id to fail.")],
    error_info: Annotated[Dict[str, object], Field(..., description="Error payload.")],
    state: Annotated[dict, InjectedState] = {},
) -> Optional[Dict[str, Any]]:
    state_api = require_state_api(state)
    task = state_api.task_fail(task_id, error_info)
    return task.to_dict() if task else None


def artifact_write(
    media_type: Annotated[str, Field(..., description="Artifact media type.")],
    body: Annotated[str, Field(..., description="Artifact body.")],
    metadata: Annotated[
        Dict[str, object], Field(..., description="Artifact metadata.")
    ],
    state: Annotated[dict, InjectedState] = {},
) -> Dict[str, Any]:
    state_api = require_state_api(state)
    artifact_id = state_api.artifact_write(media_type, body, metadata)
    return {"artifact_id": artifact_id}


def artifact_read(
    artifact_id: Annotated[str, Field(..., description="Artifact id.")],
    state: Annotated[dict, InjectedState] = {},
) -> Optional[Dict[str, Any]]:
    state_api = require_state_api(state)
    artifact = state_api.artifact_read(artifact_id)
    return artifact.to_full_dict() if artifact else None


def artifact_extract(
    artifact_id: Annotated[str, Field(..., description="Artifact id.")],
    instructions: Annotated[str, Field(..., description="Extraction instructions.")],
    state: Annotated[dict, InjectedState] = {},
) -> Dict[str, Any]:
    state_api = require_state_api(state)
    artifact = state_api.artifact_read(artifact_id)
    if not artifact:
        return {"artifact_id": artifact_id, "error": "not_found"}

    llm_api = require_tool_llm(state)

    prompt = f"extract instructions: {instructions}\n{artifact.body}"
    extracted = llm_api.generate(prompt, [])
    return {"artifact_id": artifact_id, "result": extracted}


def artifact_search(
    query: Annotated[Dict[str, object], Field(..., description="Metadata query.")],
    state: Annotated[dict, InjectedState] = {},
) -> List[Dict[str, Any]]:
    state_api = require_state_api(state)
    return [artifact.to_small_dict() for artifact in state_api.artifact_search(query)]


def artifact_list(state: Annotated[dict, InjectedState] = {}) -> List[Dict[str, Any]]:
    state_api = require_state_api(state)
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
            name="create_work",
            description=(
                "Start a worker deep-work job for investigations that may exceed the main agent’s per-step tool-call budget.\n"
                "Use when the main loop needs to offload long-running research / multi-hop browsing / heavy extraction beyond allowed tool iterations.\n"
                "Not for scheduling; for time-based or periodic runs use task.create_work_at / task.create_work_repeat.\n"
                "The worker runtime will always emit a notification at the end (not via this tool call).\n"
                "Message format must include: Purpose/Success, Constraints/Scope, Deliverable format, Required items (Conclusion/Evidence/Open items), Handling of unknowns."
            ),
        ),
        StructuredTool.from_function(
            task_create_work_at,
            name="create_work_at",
            description=(
                "Schedule a worker job at a specific run_at (ISO8601).\n"
                "Use for reminders, delayed checks, or actions that must happen at a certain time.\n"
                "Not for deep-work offloading; use task.create_work if the goal is to exceed main-loop tool budget.\n"
                "Message format must include: Purpose/Success, Constraints/Scope, Deliverable format, Required items (Conclusion/Evidence/Open items), Handling of unknowns."
            ),
        ),
        StructuredTool.from_function(
            task_create_work_repeat,
            name="create_work_repeat",
            description=(
                "Schedule a repeating worker job at a fixed interval (repeat_interval_seconds >= 3600).\n"
                "Use for periodic monitoring/digests/maintenance.\n"
                "Each run will end with a notification emitted by the worker runtime.\n"
                "Message format must include: Purpose/Success, Constraints/Scope, Deliverable format, Required items (Conclusion/Evidence/Open items), Handling of unknowns."
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
            name="get_task",
            description="Fetch a task by id for debugging, tracing, or runner logic.",
        ),
        StructuredTool.from_function(
            task_list,
            name="get_task_list",
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
        StructuredTool.from_function(
            artifact_write,
            name="write_artifact",
            description=(
                "Persist an artifact (text body + media_type + metadata) for later retrieval via semantic search and reuse across steps/workers.\n"
                "Use to store fetched web content, intermediate results, or user-requested temporary notes."
            ),
        ),
        StructuredTool.from_function(
            artifact_read,
            name="read_artifact",
            description=(
                "Read a stored artifact by id (returns full body).\n"
                "Use when you already know the exact artifact to reuse."
            ),
        ),
        StructuredTool.from_function(
            artifact_extract,
            name="extract_artifact",
            description=(
                "Run LLM-based extraction over an artifact specified by id using provided instructions (e.g., pull facts, make bullet notes, extract entities).\n"
                "Use after artifact.search when the body is long and you only need specific information; prefer this over artifact.read when possible."
            ),
        ),
        StructuredTool.from_function(
            artifact_search,
            name="search_artifact",
            description=(
                "Semantic search over stored artifacts using an embedding query against artifact bodies.\n"
                "Returns artifact IDs only (and optionally scores if available).\n"
                "Use to locate relevant artifacts, then call artifact.read to fetch the body or artifact.extract to pull targeted information."
            ),
        ),
        StructuredTool.from_function(
            artifact_list,
            name="get_list_artifact",
            description="List stored artifacts for inspection/debugging. Prefer artifact.search for finding relevant artifacts by meaning.",
        ),
    ]


def build_task_tools() -> List[BaseTool]:
    return [
        # StructuredTool.from_function(
        #     task_create_user_request,
        #     name="task.create_user_request",
        #     description="Create a user_request task.",
        # ),
        StructuredTool.from_function(
            task_create_work,
            name="create_work",
            description=(
                "Start a worker deep-work job for investigations that may exceed the main agent’s per-step tool-call budget.\n"
                "Use when the main loop needs to offload long-running research / multi-hop browsing / heavy extraction beyond allowed tool iterations.\n"
                "Not for scheduling; for time-based or periodic runs use task.create_work_at / task.create_work_repeat.\n"
                "The worker runtime will always emit a notification at the end (not via this tool call).\n"
                "Message format must include: Purpose/Success, Constraints/Scope, Deliverable format, Required items (Conclusion/Evidence/Open items), Handling of unknowns."
            ),
        ),
        StructuredTool.from_function(
            task_create_work_at,
            name="create_work_at",
            description=(
                "Schedule a worker job at a specific run_at (ISO8601).\n"
                "Use for reminders, delayed checks, or actions that must happen at a certain time.\n"
                "Not for deep-work offloading; use task.create_work if the goal is to exceed main-loop tool budget.\n"
                "Message format must include: Purpose/Success, Constraints/Scope, Deliverable format, Required items (Conclusion/Evidence/Open items), Handling of unknowns."
            ),
        ),
        StructuredTool.from_function(
            task_create_work_repeat,
            name="create_work_repeat",
            description=(
                "Schedule a repeating worker job at a fixed interval (repeat_interval_seconds >= 3600).\n"
                "Use for periodic monitoring/digests/maintenance.\n"
                "Each run will end with a notification emitted by the worker runtime.\n"
                "Message format must include: Purpose/Success, Constraints/Scope, Deliverable format, Required items (Conclusion/Evidence/Open items), Handling of unknowns."
            ),
        ),
        StructuredTool.from_function(
            task_get,
            name="get_task",
            description="Fetch a task by id for debugging, tracing, or runner logic.",
        ),
        StructuredTool.from_function(
            task_list,
            name="get_task_list",
            description="List tasks by type/state for runner polling, dashboards, or maintenance.",
        ),
    ]


def _validate_run_at(run_at: str) -> None:
    try:
        validate_run_at_future(run_at)
    except ValueError as exc:
        logger.error("invalid run_at: %s", run_at)
        raise exc
