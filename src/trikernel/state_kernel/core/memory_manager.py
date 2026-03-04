from __future__ import annotations

import asyncio
from typing import Iterable, Sequence

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.store.base import BaseStore
from langmem import (
    create_memory_manager,
    create_memory_store_manager,
    create_prompt_optimizer,
)

from .ollama import load_ollama_config
from ..memory_schemas import Episode, Procedure, Triple, UserProfile
from ...utils.logging import get_logger

logger = get_logger(__name__)


class LangMemMemoryManager:
    def __init__(
        self,
        store: BaseStore,
        *,
        model: ChatOllama | None = None,
        namespace: Iterable[str] | None = None,
    ) -> None:
        self._store = store
        self._model = model or _default_model()
        base_namespace = tuple(namespace or ("memory", "{langgraph_user_id}"))
        self._extractor = create_memory_manager(
            self._model,
            schemas=[UserProfile, Triple, Episode, Procedure],
        )
        self._profile_manager = create_memory_store_manager(
            self._model,
            namespace=base_namespace + ("profile",),
            schemas=[UserProfile],
            store=self._store,
            enable_inserts=False,
        )
        self._semantic_manager = create_memory_store_manager(
            self._model,
            namespace=base_namespace + ("semantic",),
            schemas=[Triple],
            store=self._store,
            enable_inserts=True,
            enable_deletes=True,
        )
        self._episodic_manager = create_memory_store_manager(
            self._model,
            namespace=base_namespace + ("episodic",),
            schemas=[Episode],
            store=self._store,
            enable_inserts=True,
            enable_deletes=True,
        )
        self._procedural_optimizer = create_prompt_optimizer(
            self._model, kind="metaprompt"
        )
        self._procedural_prompt = "You are a helpful assistant."
        # TODO: Decide which trajectories/feedback to use and which prompt
        # sections to update (system prompt vs tool instructions), then wire the
        # optimized prompt into the runner/config at the right cadence.

    async def update(
        self,
        messages: Sequence[BaseMessage],
        *,
        conversation_id: str,
    ) -> None:
        store_loop = getattr(self._store, "loop", None)
        if store_loop is not None and store_loop is not asyncio.get_running_loop():
            future = asyncio.run_coroutine_threadsafe(
                self._update_in_store_loop(messages, conversation_id=conversation_id),
                store_loop,
            )
            await asyncio.wrap_future(future)
            return
        await self._update_in_store_loop(messages, conversation_id=conversation_id)

    async def _update_in_store_loop(
        self,
        messages: Sequence[BaseMessage],
        *,
        conversation_id: str,
    ) -> None:
        logger.info("update memory start")
        config = {"configurable": {"langgraph_user_id": conversation_id}}
        try:
            for manager in (
                self._profile_manager,
                self._semantic_manager,
                self._episodic_manager,
            ):
                await manager.ainvoke({"messages": list(messages)}, config=config)
            self._queue_procedural_update(messages)
        except Exception:
            logger.error("memory store manager failed", exc_info=True)
            try:
                await self._extractor.ainvoke(
                    {"messages": list(messages)}, config=config
                )
            except Exception:
                logger.error("memory extraction fallback failed", exc_info=True)
            self._queue_procedural_update(messages)

    def _queue_procedural_update(self, messages: Sequence[BaseMessage]) -> None:
        _ = messages
        # TODO: Use self._procedural_optimizer.invoke with trajectories + current prompt
        # to obtain an updated prompt, then persist/apply it.


def _default_model() -> ChatOllama:
    config = load_ollama_config()
    model = config.model or "llama3"
    return ChatOllama(model=model, base_url=config.base_url)
