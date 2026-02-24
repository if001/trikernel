from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, Field

Namespace = Tuple[str, ...]


class UserProfile(BaseModel):
    display_name: Optional[str] = Field(default=None, description="User display name.")
    bio: Optional[str] = Field(default=None, description="Short user bio.")
    preferences: Dict[str, str] = Field(
        default_factory=dict, description="Stable user preferences."
    )
    tags: List[str] = Field(default_factory=list, description="Tags for retrieval.")


class Triple(BaseModel):
    subject: str = Field(..., description="Entity or subject of the fact.")
    predicate: str = Field(..., description="Relation or attribute.")
    object: str = Field(..., description="Object or value of the fact.")
    source: Optional[str] = Field(default=None, description="Where this was learned.")
    tags: List[str] = Field(default_factory=list, description="Tags for retrieval.")


class Episode(BaseModel):
    summary: str = Field(..., description="Short summary of the episode.")
    context: Optional[str] = Field(default=None, description="Context or trigger.")
    outcome: Optional[str] = Field(default=None, description="Outcome or result.")
    when: Optional[str] = Field(default=None, description="ISO8601 timestamp.")


class Procedure(BaseModel):
    pattern: Literal["gradient", "prompt_memory", "metaprompt"] = Field(
        ...,
        description="Procedure pattern (e.g., gradient, prompt_memory, metaprompt).",
    )
    description: str = Field(..., description="What the procedure is for.")
    steps: List[str] = Field(default_factory=list, description="Steps to follow.")
    when: Optional[str] = Field(default=None, description="When to apply this.")


MEMORY_SCHEMAS = (UserProfile, Triple, Episode, Procedure)
