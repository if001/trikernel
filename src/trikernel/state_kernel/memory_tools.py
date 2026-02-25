from __future__ import annotations

from typing import List

from langchain_core.tools import BaseTool
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.base import BaseStore

from .memory_schemas import Episode, Namespace, Procedure, Triple, UserProfile


def build_memory_tools(store: BaseStore) -> List[BaseTool]:
    return [
        _manage_tool(
            store,
            namespace=("memory", "{langgraph_user_id}", "profile"),
            name="memory.profile.manage",
            schema=UserProfile,
        ),
        _search_tool(
            store,
            namespace=("memory", "{langgraph_user_id}", "profile"),
            name="memory.profile.search",
        ),
        _manage_tool(
            store,
            namespace=("memory", "{langgraph_user_id}", "semantic"),
            name="memory.semantic.manage",
            schema=Triple,
        ),
        _search_tool(
            store,
            namespace=("memory", "{langgraph_user_id}", "semantic"),
            name="memory.semantic.search",
        ),
        _manage_tool(
            store,
            namespace=("memory", "{langgraph_user_id}", "episodic"),
            name="memory.episode.manage",
            schema=Episode,
        ),
        _search_tool(
            store,
            namespace=("memory", "{langgraph_user_id}", "episodic"),
            name="memory.episode.search",
        ),
        _manage_tool(
            store,
            namespace=("memory", "{langgraph_user_id}", "procedural"),
            name="memory.procedure.manage",
            schema=Procedure,
        ),
        _search_tool(
            store,
            namespace=("memory", "{langgraph_user_id}", "procedural"),
            name="memory.procedure.search",
        ),
    ]


def _search_tool(
    store: BaseStore,
    *,
    namespace: Namespace,
    name: str,
) -> BaseTool:
    return create_search_memory_tool(
        namespace=namespace,
        store=store,
        name=name,
    )


def _manage_tool(
    store: BaseStore,
    *,
    namespace: Namespace,
    name: str,
    schema: type,
) -> BaseTool:
    return create_manage_memory_tool(
        namespace=namespace,
        store=store,
        schema=schema,
        name=name,
    )
