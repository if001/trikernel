from __future__ import annotations

from typing import Sequence, Type

from langgraph.store.base import BaseStore

from ..utils.logging import get_logger
from .memory_schemas import Episode, Procedure, Triple, UserProfile

logger = get_logger(__name__)


class MemoryKernel:
    def __init__(self, store: BaseStore, conversation_id: str) -> None:
        self._store = store
        self._conversation_id = conversation_id

    def get_profile_context(
        self,
        *,
        limit: int = 3,
    ) -> str:
        try:
            profile_items = self._store.search(
                ("memory", self._conversation_id, "profile"),
                limit=limit,
            )
        except Exception:
            logger.error("failed to load profile memory context", exc_info=True)
            return ""

        if not profile_items:
            return ""
        return "Profile:\n" + _format_memory_items(profile_items)

    def get_semantic_context(
        self,
        query: str,
        *,
        limit: int = 3,
    ) -> str:
        if not query:
            return ""
        try:
            semantic_items = self._store.search(
                ("memory", self._conversation_id, "semantic"),
                query=query,
                limit=limit,
            )
        except Exception:
            logger.error("failed to load semantic memory context", exc_info=True)
            return ""
        if not semantic_items:
            return ""
        return "Semantic:\n" + _format_memory_items(semantic_items, True)

    def get_episodic_context(
        self,
        query: str,
        *,
        limit: int = 3,
    ) -> str:
        if not query:
            return ""
        try:
            episodic_items = self._store.search(
                ("memory", self._conversation_id, "episodic"),
                query=query,
                limit=limit,
            )
        except Exception:
            logger.error("failed to load episodic memory context", exc_info=True)
            return ""
        if not episodic_items:
            return ""
        return "Episodic:\n" + _format_memory_items(episodic_items, True)


def _format_memory_items(items: Sequence[object], include_score: bool = False) -> str:
    lines: list[str] = []
    for item in items:
        line = _format_memory_item(item, include_score)
        if line:
            lines.append(f"- {line}")
    return "\n".join(lines)


def _format_memory_item(item: object, include_score: bool) -> str:
    value = getattr(item, "value", None)
    key = getattr(item, "key", None)
    score = getattr(item, "score", None)
    header_parts: list[str] = []
    if key:
        header_parts.append(f"key={key}")
    if include_score and score is not None:
        header_parts.append(f"score={score:.3f}")
    header = " ".join(header_parts) if header_parts else "item"

    if isinstance(value, dict):
        details = _format_content_lines(value)
        if details:
            return header + "\n  - " + "\n  - ".join(details)
        return header
    if value is not None:
        return header + f"\n  - value: {_stringify_value(value)}"
    return header


def _format_content_lines(value: dict) -> list[str]:
    content = value.get("content") if isinstance(value, dict) else None
    kind = value.get("kind") if isinstance(value, dict) else None
    if isinstance(content, dict):
        formatted = _format_schema_content(kind, content)
        if formatted:
            return formatted.splitlines()
        return _format_kv_lines(content)
    return _format_kv_lines(value)


def _format_schema_content(kind: object, content: dict) -> str:
    model_type: Type[UserProfile | Triple | Episode | Procedure] | None = None
    if isinstance(kind, str):
        kind_lower = kind.lower()
        if "profile" in kind_lower:
            model_type = UserProfile
        elif "semantic" in kind_lower or "triple" in kind_lower:
            model_type = Triple
        elif "episode" in kind_lower or "episodic" in kind_lower:
            model_type = Episode
        elif "procedure" in kind_lower:
            model_type = Procedure
    if model_type is None:
        model_type = _infer_schema(content)
    if model_type is None:
        return ""
    try:
        model = model_type(**content)
    except Exception:
        return ""
    return model.format_markdown()


def _infer_schema(content: dict) -> Type[UserProfile | Triple | Episode | Procedure] | None:
    keys = set(content.keys())
    if {"subject", "predicate", "object"} <= keys:
        return Triple
    if {"summary"} <= keys:
        return Episode
    if {"pattern", "description"} <= keys:
        return Procedure
    if keys & {"display_name", "bio", "preferences", "tags"}:
        return UserProfile
    return None


def _format_kv_lines(value: dict) -> list[str]:
    lines: list[str] = []
    for k, v in value.items():
        lines.append(f"{k}: {_stringify_value(v)}")
    return lines


def _stringify_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(_stringify_value(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={_stringify_value(v)}" for k, v in value.items())
    if value is None:
        return "null"
    return str(value)
