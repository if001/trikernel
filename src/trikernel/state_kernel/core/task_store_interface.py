from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from ..models import Task, TaskType


class TaskStoreProtocol(Protocol):
    def create(self, task_type: TaskType, payload: Dict[str, Any]) -> Task: ...

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

