from __future__ import annotations

import json
from typing import Sequence

from langgraph.store.base import BaseStore

from ..utils.logging import get_logger

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
    if isinstance(value, dict):
        payload: dict[str, object] = {"value": value}
        if key:
            payload["key"] = key
        if include_score and score is not None:
            payload["score"] = score
        return json.dumps(payload, ensure_ascii=False)
    if value is not None:
        return str(value)
    return ""
