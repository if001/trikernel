from __future__ import annotations


from pathlib import Path
from typing import Any, Dict, List, Optional

from .file_store import JsonFileArtifactStore, JsonFileTaskStore, JsonFileTurnStore
from .models import Artifact, Task, Turn
from .protocols import ArtifactStore, StateKernelAPI, TaskStore, TurnStore


class StateKernel(StateKernelAPI):
    def __init__(
        self,
        task_store: Optional[TaskStore] = None,
        artifact_store: Optional[ArtifactStore] = None,
        turn_store: Optional[TurnStore] = None,
        data_dir: Optional[Path] = None,
    ) -> None:
        if data_dir is None:
            data_dir = Path(".state")
        self._task_store = task_store or JsonFileTaskStore(data_dir)
        self._artifact_store = artifact_store or JsonFileArtifactStore(data_dir)
        self._turn_store = turn_store or JsonFileTurnStore(data_dir)

    def task_create(self, task_type: str, payload: Dict[str, Any]) -> str:
        print(f"task creat: {task_type}, {payload}")
        return self._task_store.create(task_type, payload).task_id

    def task_get(self, task_id: str) -> Optional[Task]:
        return self._task_store.get(task_id)

    def task_update(self, task_id: str, patch: Dict[str, Any]) -> Optional[Task]:
        return self._task_store.update(task_id, patch)

    def task_list(
        self, task_type: Optional[str] = None, state: Optional[str] = None
    ) -> List[Task]:
        return self._task_store.list(task_type=task_type, state=state)

    def task_claim(
        self,
        filter_by: Dict[str, Any],
        claimer_id: str,
        ttl_seconds: int,
    ) -> Optional[str]:
        task = self._task_store.claim(filter_by, claimer_id, ttl_seconds)
        return task.task_id if task else None

    def task_complete(self, task_id: str) -> Optional[Task]:
        return self._task_store.complete(task_id)

    def task_fail(self, task_id: str, error_info: Dict[str, Any]) -> Optional[Task]:
        return self._task_store.fail(task_id, error_info)

    def artifact_write(
        self, media_type: str, body: str, metadata: Dict[str, Any]
    ) -> str:
        return self._artifact_store.write(media_type, body, metadata).artifact_id

    def artifact_read(self, artifact_id: str) -> Optional[Artifact]:
        return self._artifact_store.read(artifact_id)

    def artifact_write_named(
        self, artifact_id: str, media_type: str, body: str, metadata: Dict[str, Any]
    ) -> str:
        return self._artifact_store.write_named(
            artifact_id, media_type, body, metadata
        ).artifact_id

    def artifact_search(self, query: Dict[str, Any]) -> List[Artifact]:
        return list(self._artifact_store.search(query))

    def turn_append_user(
        self,
        conversation_id: str,
        user_message: str,
        related_task_id: Optional[str],
    ) -> str:
        return self._turn_store.append_user(
            conversation_id, user_message, related_task_id
        ).turn_id

    def turn_set_assistant(
        self,
        turn_id: str,
        assistant_message: str,
        artifacts: List[str],
        metadata: Dict[str, Any],
    ) -> Optional[Turn]:
        return self._turn_store.set_assistant(
            turn_id, assistant_message, artifacts, metadata
        )

    def turn_list_recent(self, conversation_id: str, limit: int) -> List[Turn]:
        return self._turn_store.list_recent(conversation_id, limit)
