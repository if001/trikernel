from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from ..utils.time_utils import now_iso


def utc_now() -> str:
    return now_iso()


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value)


TaskType = Literal[
    "user_request",
    "work",
    "notification",
    "tool_loop.step",
    "tool_loop.followup",
    "tool_loop.final",
]


@dataclass
class Task:
    task_id: str
    task_type: TaskType
    payload: Dict[str, Any]
    state: str
    artifact_refs: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    claimed_by: Optional[str] = None
    claim_expires_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "state": self.state,
            "artifact_refs": list(self.artifact_refs),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "claimed_by": self.claimed_by,
            "claim_expires_at": self.claim_expires_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            payload=data.get("payload", {}),
            state=data["state"],
            artifact_refs=list(data.get("artifact_refs", [])),
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
            claimed_by=data.get("claimed_by"),
            claim_expires_at=data.get("claim_expires_at"),
        )


@dataclass
class Artifact:
    artifact_id: str
    media_type: str
    body: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "media_type": self.media_type,
            "body": self.body,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        return cls(
            artifact_id=data["artifact_id"],
            media_type=data["media_type"],
            body=data.get("body", ""),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", utc_now()),
        )
