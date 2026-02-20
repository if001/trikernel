from __future__ import annotations

import json
import os
import threading
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from .models import Artifact, Task, Turn, parse_time, utc_now
from ..utils.search import HybridSearchIndex


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
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
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

    def list(
        self,
        task_type: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[Task]:
        with self._lock:
            data = self._read_all()
        tasks = [Task.from_dict(value) for value in data.values()]
        if task_type is None and state is None:
            return tasks
        result = []
        for task in tasks:
            if task_type is not None and task.task_type != task_type:
                continue
            if state is not None and task.state != state:
                continue
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
        self._artifact_dir = data_dir / "artifacts"
        self._lock = threading.Lock()
        data_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._search_index = _init_artifact_search(data_dir)
        self._rebuild_index()

    def write(self, media_type: str, body: str, metadata: Dict[str, Any]) -> Artifact:
        with self._lock:
            artifact_id = str(uuid4())
            artifact = Artifact(
                artifact_id=artifact_id,
                media_type=media_type,
                body=body,
                metadata=metadata,
            )
            self._write_file(artifact)
            self._index_artifact(artifact)
            return artifact

    def read(self, artifact_id: str) -> Optional[Artifact]:
        with self._lock:
            path = self._artifact_path(artifact_id)
            if not path.exists():
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
            return Artifact.from_dict(raw)

    def write_named(
        self, artifact_id: str, media_type: str, body: str, metadata: Dict[str, Any]
    ) -> Artifact:
        with self._lock:
            artifact = Artifact(
                artifact_id=artifact_id,
                media_type=media_type,
                body=body,
                metadata=metadata,
            )
            self._write_file(artifact)
            self._index_artifact(artifact)
            return artifact

    def search(self, query: Dict[str, Any]) -> Iterable[Artifact]:
        with self._lock:
            return self._search_locked(query)

    def _artifact_path(self, artifact_id: str) -> Path:
        return self._artifact_dir / f"{artifact_id}.json"

    def _write_file(self, artifact: Artifact) -> None:
        path = self._artifact_path(artifact.artifact_id)
        path.write_text(
            json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _index_artifact(self, artifact: Artifact) -> None:
        metadata = _normalize_metadata(artifact.metadata)
        metadata["artifact_id"] = artifact.artifact_id
        metadata["media_type"] = artifact.media_type
        metadata["id"] = artifact.artifact_id
        doc = Document(page_content=artifact.body, metadata=metadata)
        self._search_index.upsert_document(doc, artifact.artifact_id, force=True)

    def _search_locked(self, query: Dict[str, Any]) -> List[Artifact]:
        text_query = str(query.get("text") or query.get("query") or "").strip()
        limit = int(query.get("k") or query.get("limit") or 5)
        metadata_filter = query.get("metadata")
        if text_query:
            docs = self._search_index.search(
                text_query, k=limit, metadata_filter=metadata_filter
            )
            return [
                artifact
                for artifact in (
                    self._read_by_id(doc.metadata.get("artifact_id")) for doc in docs
                )
                if artifact
            ]
        artifacts = self._all_artifacts()
        if not query:
            return artifacts
        result = []
        for artifact in artifacts:
            if _matches_query(artifact, query):
                result.append(artifact)
        return result

    def _read_by_id(self, artifact_id: Optional[str]) -> Optional[Artifact]:
        if not artifact_id:
            return None
        path = self._artifact_path(str(artifact_id))
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Artifact.from_dict(raw)

    def _all_artifacts(self) -> List[Artifact]:
        return [
            artifact
            for artifact in (
                self._read_by_id(path.stem)
                for path in self._artifact_dir.glob("*.json")
            )
            if artifact
        ]

    def _rebuild_index(self) -> None:
        docs = []
        for artifact in self._all_artifacts():
            metadata = _normalize_metadata(artifact.metadata)
            metadata["artifact_id"] = artifact.artifact_id
            metadata["media_type"] = artifact.media_type
            metadata["id"] = artifact.artifact_id
            docs.append(Document(page_content=artifact.body, metadata=metadata))
        self._search_index.set_documents(docs)


def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            normalized[key] = value
        else:
            normalized[key] = json.dumps(value, ensure_ascii=False)
    return normalized


def _matches_query(artifact: Artifact, query: Dict[str, Any]) -> bool:
    for key, value in query.items():
        if key in {"text", "query", "k", "limit"}:
            continue
        if key == "metadata" and isinstance(value, dict):
            for meta_key, meta_value in value.items():
                if artifact.metadata.get(meta_key) != meta_value:
                    return False
            continue
        if getattr(artifact, key, None) != value:
            return False
    return True


def _init_artifact_search(data_dir: Path) -> HybridSearchIndex:
    load_dotenv()
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
    persist_dir = data_dir / "search_artifacts"
    return HybridSearchIndex(persist_dir, "artifacts", embeddings)


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
