from __future__ import annotations

import json
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import InjectedState
from pydantic import Field
from typing_extensions import Annotated

from trikernel.utils.logging import get_logger

from .prompts import (
    build_article_prompt,
    build_extract_prompt,
    build_outline_prompt,
    build_polish_prompt,
    build_summary_prompt,
)
from ._shared import require_tool_llm

logger = get_logger(__name__)




def summarize_text(
    text: Annotated[str, Field(..., description="Text to summarize.")],
    max_length: Annotated[
        Optional[int], Field(default=None, description="Max length.")
    ] = None,
    style: Annotated[
        Optional[str], Field(default=None, description="Summary style.")
    ] = None,
    language: Annotated[
        Optional[str], Field(default="Japanese", description="Output language.")
    ] = "Japanese",
    state: Annotated[dict, InjectedState] = {},
) -> Dict[str, object]:
    llm_api = require_tool_llm(state)
    prompt = build_summary_prompt(
        text=text,
        max_length=max_length,
        style=style,
        language=language,
    )
    response_text = llm_api.generate(prompt, [])
    return _parse_json_or_text(response_text)


def extract_corresponding(
    source_text: Annotated[str, Field(..., description="Reference/source text.")],
    target_text: Annotated[str, Field(..., description="Target text to extract from.")],
    criteria: Annotated[
        Optional[str], Field(default=None, description="Selection criteria.")
    ] = None,
    language: Annotated[
        Optional[str], Field(default="Japanese", description="Output language.")
    ] = "Japanese",
    state: Annotated[dict, InjectedState] = {},
) -> Dict[str, object]:
    llm_api = require_tool_llm(state)
    prompt = build_extract_prompt(
        source_text=source_text,
        target_text=target_text,
        criteria=criteria,
        language=language,
    )
    response_text = llm_api.generate(prompt, [])
    return _parse_json_or_text(response_text)


def create_outline(
    user_input: Annotated[
        Optional[str], Field(default=None, description="User input.")
    ] = None,
    tool_results: Annotated[
        Optional[List[str]], Field(default=None, description="Tool result summaries.")
    ] = None,
    article_type: Annotated[
        Optional[str], Field(default=None, description="Article type.")
    ] = None,
    audience: Annotated[
        Optional[str], Field(default=None, description="Target audience.")
    ] = None,
    language: Annotated[
        Optional[str], Field(default="Japanese", description="Output language.")
    ] = "Japanese",
    *,
    state: Annotated[dict, InjectedState] = {},
) -> Dict[str, object]:
    llm_api = require_tool_llm(state)
    prompt = build_outline_prompt(
        user_input=user_input,
        tool_results=tool_results,
        article_type=article_type,
        audience=audience,
        language=language,
    )
    response_text = llm_api.generate(prompt, [])
    return _parse_json_or_text(response_text)


def polish_article(
    draft: Annotated[str, Field(..., description="Article draft.")],
    article_type: Annotated[
        Optional[str], Field(default=None, description="Article type.")
    ] = None,
    audience: Annotated[
        Optional[str], Field(default=None, description="Target audience.")
    ] = None,
    language: Annotated[
        Optional[str], Field(default="Japanese", description="Output language.")
    ] = "Japanese",
    state: Annotated[dict, InjectedState] = {},
) -> Dict[str, object]:
    llm_api = require_tool_llm(state)
    prompt = build_polish_prompt(
        draft=draft,
        article_type=article_type,
        audience=audience,
        language=language,
    )
    response_text = llm_api.generate(prompt, [])
    return _parse_json_or_text(response_text)


def generate_article(
    article_type: Annotated[str, Field(..., description="Article type.")],
    audience: Annotated[str, Field(..., description="Target audience.")],
    draft: Annotated[str, Field(..., description="Draft content.")],
    revisions: Annotated[
        Optional[List[str]], Field(default=None, description="Revision points.")
    ] = None,
    outline: Annotated[
        Optional[str], Field(default=None, description="Outline content.")
    ] = None,
    language: Annotated[
        Optional[str], Field(default="Japanese", description="Output language.")
    ] = "Japanese",
    state: Annotated[dict, InjectedState] = {},
) -> Dict[str, object]:
    llm_api = require_tool_llm(state)
    prompt = build_article_prompt(
        article_type=article_type,
        audience=audience,
        revisions=revisions,
        outline=outline,
        draft=draft,
        language=language,
    )
    response_text = llm_api.generate(prompt, [])
    return _parse_json_or_text(response_text)


def build_writing_tools() -> List[BaseTool]:
    return [
        StructuredTool.from_function(
            summarize_text,
            name="text.summarize",
            description=(
                "Summarize long text into a shorter form with optional max length/style/language.\n"
                "Use to compress tool results before adding them to prompts/artifacts."
            ),
        ),
        StructuredTool.from_function(
            extract_corresponding,
            name="text.extract",
            description=(
                "Extract specific information from target_text, guided by source_text and optional criteria.\n"
                "Use when you have a reference schema/template and want structured selection."
            ),
        ),
        StructuredTool.from_function(
            create_outline,
            name="article.outline",
            description=(
                "Create an article outline from user intent and tool result summaries.\n"
                "Use when the user asks for a written deliverable (blog, doc, report)."
            ),
        ),
        StructuredTool.from_function(
            polish_article,
            name="article.polish",
            description="Improve clarity/structure/tone of an article draft for a target audience.",
        ),
        StructuredTool.from_function(
            generate_article,
            name="article.generate",
            description=(
                "Generate a full article from an outline and/or draft + revision points.\n"
                "Use for final deliverable generation, not for internal reasoning."
            ),
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
