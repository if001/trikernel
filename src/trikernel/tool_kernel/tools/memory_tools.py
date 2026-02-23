from __future__ import annotations

from typing import List, Optional, Tuple, Literal

from langchain_core.tools import BaseTool
from langmem import create_manage_memory_tool, create_search_memory_tool
from pydantic import BaseModel, Field

from ..memory_store import JsonFileMemoryStore

Namespace = Tuple[str, ...]


class SemanticMemory(BaseModel):
    fact: str = Field(..., description="A stable user fact or preference.")
    source: Optional[str] = Field(default=None, description="Where this was learned.")
    tags: List[str] = Field(default_factory=list, description="Tags for retrieval.")


class EpisodicMemory(BaseModel):
    summary: str = Field(..., description="Short summary of the episode.")
    context: Optional[str] = Field(default=None, description="Context or trigger.")
    outcome: Optional[str] = Field(default=None, description="Outcome or result.")
    when: Optional[str] = Field(default=None, description="ISO8601 timestamp.")


class ProceduralMemory(BaseModel):
    pattern: Literal["gradient", "prompt_memory", "metaprompt"] = Field(
        ...,
        description="Procedure pattern (e.g., gradient, prompt_memory, metaprompt).",
    )
    description: str = Field(..., description="What the procedure is for.")
    steps: List[str] = Field(default_factory=list, description="Steps to follow.")
    when: Optional[str] = Field(default=None, description="When to apply this.")


def build_memory_tools(store: JsonFileMemoryStore) -> List[BaseTool]:
    return [
        _manage_tool(
            store,
            namespace=("memories", "default", "semantic"),
            name="memory.semantic.manage",
            schema=SemanticMemory,
            instructions=(
                "Use for stable user facts and preferences. Create or update when new "
                "user preferences or identity facts are learned."
            ),
        ),
        _search_tool(
            store,
            namespace=("memories", "default", "semantic"),
            name="memory.semantic.search",
        ),
        _manage_tool(
            store,
            namespace=("memories", "default", "episodic"),
            name="memory.episode.manage",
            schema=EpisodicMemory,
            instructions=(
                "Use for notable events or interactions tied to time, outcomes, or context."
            ),
        ),
        _search_tool(
            store,
            namespace=("memories", "default", "episodic"),
            name="memory.episode.search",
        ),
        _manage_tool(
            store,
            namespace=("memories", "default", "procedural"),
            name="memory.procedure.manage",
            schema=ProceduralMemory,
            instructions=(
                "Use for repeatable procedures or workflow guidance. Pattern should be one "
                "of: gradient, prompt_memory, metaprompt."
            ),
        ),
        _search_tool(
            store,
            namespace=("memories", "default", "procedural"),
            name="memory.procedure.search",
        ),
    ]


def _manage_tool(
    store: JsonFileMemoryStore,
    *,
    namespace: Namespace,
    name: str,
    schema: type[BaseModel],
    instructions: str,
) -> BaseTool:
    return create_manage_memory_tool(
        namespace=namespace,
        schema=schema,
        instructions=instructions,
        store=store,
        name=name,
    )


def _search_tool(
    store: JsonFileMemoryStore,
    *,
    namespace: Namespace,
    name: str,
) -> BaseTool:
    return create_search_memory_tool(
        namespace=namespace,
        store=store,
        name=name,
    )
