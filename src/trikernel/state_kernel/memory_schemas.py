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

    def format_markdown(self) -> str:
        lines = []
        if self.display_name:
            lines.append(f"display_name: {self.display_name}")
        if self.bio:
            lines.append(f"bio: {self.bio}")
        if self.preferences:
            prefs = ", ".join(f"{k}={v}" for k, v in self.preferences.items())
            lines.append(f"preferences: {prefs}")
        if self.tags:
            lines.append(f"tags: {', '.join(self.tags)}")
        return "\n".join(lines)


class Triple(BaseModel):
    subject: str = Field(..., description="Entity or subject of the fact.")
    predicate: str = Field(..., description="Relation or attribute.")
    object: str = Field(..., description="Object or value of the fact.")
    source: Optional[str] = Field(default=None, description="Where this was learned.")
    tags: List[str] = Field(default_factory=list, description="Tags for retrieval.")

    def format_markdown(self) -> str:
        lines = [f"fact: {self.subject} {self.predicate} {self.object}"]
        if self.source:
            lines.append(f"source: {self.source}")
        if self.tags:
            lines.append(f"tags: {', '.join(self.tags)}")
        return "\n".join(lines)


class Episode(BaseModel):
    summary: str = Field(..., description="Short summary of the episode.")
    context: Optional[str] = Field(default=None, description="Context or trigger.")
    outcome: Optional[str] = Field(default=None, description="Outcome or result.")
    when: Optional[str] = Field(default=None, description="ISO8601 timestamp.")

    def format_markdown(self) -> str:
        lines = []
        if self.summary:
            lines.append(f"summary: {self.summary}")
        if self.context:
            lines.append(f"context: {self.context}")
        if self.outcome:
            lines.append(f"outcome: {self.outcome}")
        if self.when:
            lines.append(f"when: {self.when}")
        return "\n".join(lines)


class Procedure(BaseModel):
    pattern: Literal["gradient", "prompt_memory", "metaprompt"] = Field(
        ...,
        description="Procedure pattern (e.g., gradient, prompt_memory, metaprompt).",
    )
    description: str = Field(..., description="What the procedure is for.")
    steps: List[str] = Field(default_factory=list, description="Steps to follow.")
    when: Optional[str] = Field(default=None, description="When to apply this.")

    def format_markdown(self) -> str:
        lines = [f"pattern: {self.pattern}", f"description: {self.description}"]
        if self.steps:
            lines.append(f"steps: {', '.join(self.steps)}")
        if self.when:
            lines.append(f"when: {self.when}")
        return "\n".join(lines)


MEMORY_SCHEMAS = (UserProfile, Triple, Episode, Procedure)
