from __future__ import annotations

import json
import threading
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from .models import Artifact, Task, Turn, parse_time, utc_now


def _merge_patch(target: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(target)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_patch(merged[key], value)
        else:
            merged[key] = value
    return merged


class JsonFileTaskStore:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "tasks.json"
        self._lock = threading.Lock()
        data_dir.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("{}", encoding="utf-8")

    def _read_all(self) -> Dict[str, Dict[str, Any]]:
        raw = self._path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)

    def _write_all(self, data: Dict[str, Dict[str, Any]]) -> None:
        self._path.write_text(
            json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def create(self, task_type: str, payload: Dict[str, Any]) -> Task:
        with self._lock:
            data = self._read_all()
            task_id = str(uuid4())
            task = Task(
                task_id=task_id, task_type=task_type, payload=payload, state="queued"
            )
            data[task_id] = task.to_dict()
            self._write_all(data)
            return task

    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            data = self._read_all()
            raw = data.get(task_id)
            return Task.from_dict(raw) if raw else None

    def update(self, task_id: str, patch: Dict[str, Any]) -> Optional[Task]:
        with self._lock:
            data = self._read_all()
            current = data.get(task_id)
            if not current:
                return None
            updated = _merge_patch(current, patch)
            updated["updated_at"] = utc_now()
            data[task_id] = updated
            self._write_all(data)
            return Task.from_dict(updated)

    def list(self, filter_by: Optional[Dict[str, Any]] = None) -> List[Task]:
        with self._lock:
            data = self._read_all()
        tasks = [Task.from_dict(value) for value in data.values()]
        if not filter_by:
            return tasks
        result = []
        for task in tasks:
            matched = True
            for key, value in filter_by.items():
                if getattr(task, key, None) != value:
                    matched = False
                    break
            if matched:
                result.append(task)
        return result

    def claim(
        self,
        filter_by: Dict[str, Any],
        claimer_id: str,
        ttl_seconds: int,
    ) -> Optional[Task]:
        with self._lock:
            data = self._read_all()
            now = parse_time(utc_now())
            for task_id, raw in data.items():
                task = Task.from_dict(raw)
                if task.state not in {"queued", "running"}:
                    continue
                matched = True
                for key, value in filter_by.items():
                    if getattr(task, key, None) != value:
                        matched = False
                        break
                if not matched:
                    continue
                expires = parse_time(task.claim_expires_at)
                if task.claimed_by and expires and now and expires > now:
                    continue
                claim_expires_at = (
                    (now + timedelta(seconds=ttl_seconds)).isoformat() if now else None
                )
                updated = replace(
                    task,
                    claimed_by=claimer_id,
                    claim_expires_at=claim_expires_at,
                    state="running",
                    updated_at=utc_now(),
                )
                data[task_id] = updated.to_dict()
                self._write_all(data)
                return updated
        return None

    def complete(self, task_id: str) -> Optional[Task]:
        return self.update(
            task_id, {"state": "done", "claimed_by": None, "claim_expires_at": None}
        )

    def fail(self, task_id: str, error_info: Dict[str, Any]) -> Optional[Task]:
        return self.update(
            task_id,
            {
                "state": "failed",
                "payload": {"error": error_info},
                "claimed_by": None,
                "claim_expires_at": None,
            },
        )


class JsonFileArtifactStore:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "artifacts.json"
        self._lock = threading.Lock()
        data_dir.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("{}", encoding="utf-8")

    def _read_all(self) -> Dict[str, Dict[str, Any]]:
        raw = self._path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)

    def _write_all(self, data: Dict[str, Dict[str, Any]]) -> None:
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def write(self, media_type: str, body: str, metadata: Dict[str, Any]) -> Artifact:
        with self._lock:
            data = self._read_all()
            artifact_id = str(uuid4())
            artifact = Artifact(
                artifact_id=artifact_id,
                media_type=media_type,
                body=body,
                metadata=metadata,
            )
            data[artifact_id] = artifact.to_dict()
            self._write_all(data)
            return artifact

    def read(self, artifact_id: str) -> Optional[Artifact]:
        with self._lock:
            data = self._read_all()
            raw = data.get(artifact_id)
            return Artifact.from_dict(raw) if raw else None

    def write_named(
        self, artifact_id: str, media_type: str, body: str, metadata: Dict[str, Any]
    ) -> Artifact:
        with self._lock:
            data = self._read_all()
            artifact = Artifact(
                artifact_id=artifact_id,
                media_type=media_type,
                body=body,
                metadata=metadata,
            )
            data[artifact_id] = artifact.to_dict()
            self._write_all(data)
            return artifact

    def search(self, query: Dict[str, Any]) -> Iterable[Artifact]:
        with self._lock:
            data = self._read_all()
        artifacts = [Artifact.from_dict(value) for value in data.values()]
        if not query:
            return artifacts
        result = []
        for artifact in artifacts:
            matched = True
            for key, value in query.items():
                if getattr(artifact, key, None) != value:
                    matched = False
                    break
            if matched:
                result.append(artifact)
        return result


class JsonFileTurnStore:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "turns.json"
        self._lock = threading.Lock()
        data_dir.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("{}", encoding="utf-8")

    def _read_all(self) -> Dict[str, Dict[str, Any]]:
        raw = self._path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)

    def _write_all(self, data: Dict[str, Dict[str, Any]]) -> None:
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def append_user(
        self, conversation_id: str, user_message: str, related_task_id: Optional[str]
    ) -> Turn:
        with self._lock:
            data = self._read_all()
            turn_id = str(uuid4())
            turn = Turn(
                turn_id=turn_id,
                conversation_id=conversation_id,
                user_message=user_message,
                related_task_id=related_task_id,
            )
            data[turn_id] = turn.to_dict()
            self._write_all(data)
            return turn

    def set_assistant(
        self,
        turn_id: str,
        assistant_message: str,
        artifacts: List[str],
        metadata: Dict[str, Any],
    ) -> Optional[Turn]:
        with self._lock:
            data = self._read_all()
            current = data.get(turn_id)
            if not current:
                return None
            updated = dict(current)
            updated["assistant_message"] = assistant_message
            updated["artifacts"] = list(artifacts)
            updated["metadata"] = dict(metadata)
            updated["updated_at"] = utc_now()
            data[turn_id] = updated
            self._write_all(data)
            return Turn.from_dict(updated)

    def list_recent(self, conversation_id: str, limit: int) -> List[Turn]:
        with self._lock:
            data = self._read_all()
        turns = [
            Turn.from_dict(value)
            for value in data.values()
            if value.get("conversation_id") == conversation_id
        ]
        turns.sort(key=lambda item: item.created_at, reverse=True)
        return list(reversed(turns[:limit]))
