from __future__ import annotations


from pathlib import Path
from typing import Any, Dict, List, Optional

from trikernel.state_kernel.protocols import StateKernelAPI
from trikernel.utils.logging import get_logger

from .file_store_impl import JsonFileArtifactStore, JsonFileTaskStore
from .artifact_store_interface import ArtifactStoreProtocol
from .task_store_interface import TaskStoreProtocol
from .memory_kernel import MemoryKernel
from ..models import Artifact, Task, TaskType
from langgraph.store.base import BaseStore

logger = get_logger(__name__)


class StateKernel(StateKernelAPI):
    def __init__(
        self,
        task_store: Optional[TaskStoreProtocol] = None,
        artifact_store: Optional[ArtifactStoreProtocol] = None,
        data_dir: Optional[Path] = None,
        memory_store: Optional[BaseStore] = None,
    ) -> None:
        if data_dir is None:
            data_dir = Path(".state")
        self._task_store = task_store or JsonFileTaskStore(data_dir)
        self._artifact_store = artifact_store or JsonFileArtifactStore(data_dir)
        self._memory_store = memory_store

    def task_create(self, task_type: TaskType, payload: Dict[str, Any]) -> str:
        logger.info(f"task_create: {task_type}, {payload}")
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

    def get_artifact_path(self, id: str) -> str:
        return str(self._artifact_store.artifact_path(id))

    def artifact_read(self, artifact_id: str) -> Optional[Artifact]:
        return self._artifact_store.read(artifact_id)

    def artifact_write_named(
        self, artifact_id: str, media_type: str, body: str, metadata: Dict[str, Any]
    ) -> str:
        return self._artifact_store.write_named(
            artifact_id, media_type, body, metadata
        ).artifact_id

    def artifact_list(self) -> List[Artifact]:
        return list(self._artifact_store.list())

    def artifact_search(self, query: Dict[str, Any]) -> List[Artifact]:
        return list(self._artifact_store.search(query))

    def set_memory_store(self, store: BaseStore) -> None:
        self._memory_store = store

    def memory_kernel(self, conversation_id: str) -> MemoryKernel | None:
        if self._memory_store is None:
            raise ValueError("memory kernel not set")
        return MemoryKernel(self._memory_store, conversation_id)
