from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value)


TaskType = Literal[
    "user_request",
    "work",
    "notification",
    "pdca.plan",
    "pdca.do",
    "pdca.do.followup",
    "pdca.check",
    "pdca.discover",
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


@dataclass
class Turn:
    turn_id: str
    conversation_id: str
    user_message: str
    assistant_message: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_task_id: Optional[str] = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "artifacts": list(self.artifacts),
            "metadata": self.metadata,
            "related_task_id": self.related_task_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        return cls(
            turn_id=data["turn_id"],
            conversation_id=data["conversation_id"],
            user_message=data.get("user_message", ""),
            assistant_message=data.get("assistant_message"),
            artifacts=list(data.get("artifacts", [])),
            metadata=data.get("metadata", {}),
            related_task_id=data.get("related_task_id"),
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
        )
