from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Protocol, TypeAlias

from langgraph.checkpoint.base import BaseCheckpointSaver

Checkpointer = BaseCheckpointSaver

from .models import Artifact, Task, TaskType
from .memory_kernel import MemoryKernel


class TaskStore(Protocol):
    def create(self, task_type: str, payload: Dict[str, Any]) -> Task: ...

    def get(self, task_id: str) -> Optional[Task]: ...

    def update(self, task_id: str, patch: Dict[str, Any]) -> Optional[Task]: ...

    def list(
        self,
        task_type: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[Task]: ...

    def claim(
        self,
        filter_by: Dict[str, Any],
        claimer_id: str,
        ttl_seconds: int,
    ) -> Optional[Task]: ...

    def complete(self, task_id: str) -> Optional[Task]: ...

    def fail(self, task_id: str, error_info: Dict[str, Any]) -> Optional[Task]: ...


class ArtifactStore(Protocol):
    def write(
        self, media_type: str, body: str, metadata: Dict[str, Any]
    ) -> Artifact: ...

    def read(self, artifact_id: str) -> Optional[Artifact]: ...

    def write_named(
        self, artifact_id: str, media_type: str, body: str, metadata: Dict[str, Any]
    ) -> Artifact: ...

    def list(self) -> List[Artifact]: ...

    def search(self, query: Dict[str, Any]) -> Iterable[Artifact]: ...


class MessageStoreAPI(Protocol):
    checkpointer: Checkpointer


class StateKernelAPI(Protocol):
    def task_create(self, task_type: TaskType, payload: Dict[str, Any]) -> str: ...

    def task_get(self, task_id: str) -> Optional[Task]: ...

    def task_update(self, task_id: str, patch: Dict[str, Any]) -> Optional[Task]: ...

    def task_list(
        self,
        task_type: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[Task]: ...

    def task_claim(
        self,
        filter_by: Dict[str, Any],
        claimer_id: str,
        ttl_seconds: int,
    ) -> Optional[str]: ...

    def task_complete(self, task_id: str) -> Optional[Task]: ...

    def task_fail(self, task_id: str, error_info: Dict[str, Any]) -> Optional[Task]: ...

    def artifact_write(
        self, media_type: str, body: str, metadata: Dict[str, Any]
    ) -> str: ...

    def get_artifact_path(self, id: str) -> str: ...

    def artifact_read(self, artifact_id: str) -> Optional[Artifact]: ...

    def artifact_write_named(
        self, artifact_id: str, media_type: str, body: str, metadata: Dict[str, Any]
    ) -> str: ...

    def artifact_list(self) -> List[Artifact]: ...

    def artifact_search(self, query: Dict[str, Any]) -> List[Artifact]: ...

    def memory_kernel(self, conversation_id: str) -> MemoryKernel | None: ...
