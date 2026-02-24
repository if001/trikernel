from __future__ import annotations

import json
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from trikernel.utils.logging import get_logger

from .writing_prompts import (
    build_article_prompt,
    build_extract_prompt,
    build_outline_prompt,
    build_polish_prompt,
    build_summary_prompt,
)
from ..runtime import get_llm_api
from ..protocols import ToolLLMBase

logger = get_logger(__name__)


def _require_llm(state: dict) -> ToolLLMBase:
    llm_api = state.get("llm_api") if isinstance(state, dict) else None
    if llm_api is None and isinstance(state, dict):
        runtime_id = state.get("runtime_id")
        if isinstance(runtime_id, str):
            llm_api = get_llm_api(runtime_id)
    if llm_api is None:
        raise ValueError("llm_api is required in state")
    return llm_api


class SummarizeArgs(BaseModel):
    text: str = Field(..., description="Text to summarize.")
    max_length: Optional[int] = Field(default=None, description="Max length.")
    style: Optional[str] = Field(default=None, description="Summary style.")
    language: Optional[str] = Field(default="Japanese", description="Output language.")


class ExtractArgs(BaseModel):
    source_text: str = Field(..., description="Reference/source text.")
    target_text: str = Field(..., description="Target text to extract from.")
    criteria: Optional[str] = Field(default=None, description="Selection criteria.")
    language: Optional[str] = Field(default="Japanese", description="Output language.")


class OutlineArgs(BaseModel):
    user_input: Optional[str] = Field(default=None, description="User input.")
    tool_results: Optional[List[str]] = Field(
        default=None, description="Tool result summaries."
    )
    article_type: Optional[str] = Field(default=None, description="Article type.")
    audience: Optional[str] = Field(default=None, description="Target audience.")
    language: Optional[str] = Field(default="Japanese", description="Output language.")


class PolishArgs(BaseModel):
    draft: str = Field(..., description="Article draft.")
    article_type: Optional[str] = Field(default=None, description="Article type.")
    audience: Optional[str] = Field(default=None, description="Target audience.")
    language: Optional[str] = Field(default="Japanese", description="Output language.")


class GenerateArgs(BaseModel):
    article_type: str = Field(..., description="Article type.")
    audience: str = Field(..., description="Target audience.")
    revisions: Optional[List[str]] = Field(
        default=None, description="Revision points."
    )
    outline: Optional[str] = Field(default=None, description="Outline content.")
    draft: str = Field(..., description="Draft content.")
    language: Optional[str] = Field(default="Japanese", description="Output language.")


def summarize_text(
    payload: SummarizeArgs,
    state: Annotated[dict, InjectedState],
) -> Dict[str, object]:
    llm_api = _require_llm(state)
    prompt = build_summary_prompt(
        text=payload.text,
        max_length=payload.max_length,
        style=payload.style,
        language=payload.language,
    )
    response_text = llm_api.generate(prompt, [])
    return _parse_json_or_text(response_text)


def extract_corresponding(
    payload: ExtractArgs,
    state: Annotated[dict, InjectedState],
) -> Dict[str, object]:
    llm_api = _require_llm(state)
    prompt = build_extract_prompt(
        source_text=payload.source_text,
        target_text=payload.target_text,
        criteria=payload.criteria,
        language=payload.language,
    )
    response_text = llm_api.generate(prompt, [])
    return _parse_json_or_text(response_text)


def create_outline(
    payload: OutlineArgs,
    state: Annotated[dict, InjectedState],
) -> Dict[str, object]:
    llm_api = _require_llm(state)
    prompt = build_outline_prompt(
        user_input=payload.user_input,
        tool_results=payload.tool_results,
        article_type=payload.article_type,
        audience=payload.audience,
        language=payload.language,
    )
    response_text = llm_api.generate(prompt, [])
    return _parse_json_or_text(response_text)


def polish_article(
    payload: PolishArgs,
    state: Annotated[dict, InjectedState],
) -> Dict[str, object]:
    llm_api = _require_llm(state)
    prompt = build_polish_prompt(
        draft=payload.draft,
        article_type=payload.article_type,
        audience=payload.audience,
        language=payload.language,
    )
    response_text = llm_api.generate(prompt, [])
    return _parse_json_or_text(response_text)


def generate_article(
    payload: GenerateArgs,
    state: Annotated[dict, InjectedState],
) -> Dict[str, object]:
    llm_api = _require_llm(state)
    prompt = build_article_prompt(
        article_type=payload.article_type,
        audience=payload.audience,
        revisions=payload.revisions,
        outline=payload.outline,
        draft=payload.draft,
        language=payload.language,
    )
    response_text = llm_api.generate(prompt, [])
    return _parse_json_or_text(response_text)


def build_writing_tools() -> List[BaseTool]:
    return [
        StructuredTool.from_function(
            summarize_text,
            name="text.summarize",
            description="Summarize a given text.",
        ),
        StructuredTool.from_function(
            extract_corresponding,
            name="text.extract",
            description="Extract information from text based on criteria.",
        ),
        StructuredTool.from_function(
            create_outline,
            name="article.outline",
            description="Create an outline for an article.",
        ),
        StructuredTool.from_function(
            polish_article,
            name="article.polish",
            description="Polish an article draft.",
        ),
        StructuredTool.from_function(
            generate_article,
            name="article.generate",
            description="Generate an article from an outline or draft.",
        ),
    ]


def _parse_json_or_text(text: str) -> Dict[str, object]:
    if not text:
        return {"result": ""}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"result": text}
    if isinstance(parsed, dict):
        return parsed
    return {"result": parsed}
