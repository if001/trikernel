from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional
from trikernel.utils.logging import get_logger

from ..models import ToolContext
from ..protocols import ToolLLMAPI
from .writing_prompts import (
    build_article_prompt,
    build_extract_prompt,
    build_outline_prompt,
    build_polish_prompt,
    build_summary_prompt,
)

logger = get_logger(__name__)


def _require_llm(context: ToolContext) -> ToolLLMAPI:
    if context is None or context.llm_api is None:
        raise ValueError("llm_api is required in ToolContext")
    return context.llm_api


def _parse_json_or_text(text: str) -> Dict[str, object]:
    if not text:
        return {"error": "empty_response"}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def summarize_text(
    text: str,
    max_length: Optional[int] = None,
    style: Optional[str] = None,
    language: Optional[str] = "ja",
    *,
    context: ToolContext,
) -> Dict[str, object]:
    llm_api = _require_llm(context)
    prompt = build_summary_prompt(text, max_length, style, language)
    response_text = llm_api.generate(prompt, [])
    return _parse_json_or_text(response_text)


def extract_corresponding(
    source_text: str,
    target_text: str,
    criteria: Optional[str] = None,
    language: Optional[str] = "ja",
    *,
    context: ToolContext,
) -> Dict[str, object]:
    llm_api = _require_llm(context)
    prompt = build_extract_prompt(source_text, target_text, criteria, language)
    response_text = llm_api.generate(prompt, [])
    return _parse_json_or_text(response_text)


def create_outline(
    user_input: Optional[str] = None,
    tool_results: Optional[List[str]] = None,
    article_type: Optional[str] = None,
    audience: Optional[str] = None,
    language: Optional[str] = "ja",
    *,
    context: ToolContext,
) -> Dict[str, object]:
    llm_api = _require_llm(context)
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
    draft: str,
    article_type: Optional[str] = None,
    audience: Optional[str] = None,
    language: Optional[str] = "ja",
    *,
    context: ToolContext,
) -> Dict[str, object]:
    llm_api = _require_llm(context)
    prompt = build_polish_prompt(draft, article_type, audience, language)
    response_text = llm_api.generate(prompt, [])
    return _parse_json_or_text(response_text)


def generate_article(
    article_type: str,
    audience: str,
    revisions: Optional[List[str]] = None,
    outline: Optional[str] = None,
    draft: str = "",
    language: Optional[str] = "ja",
    *,
    context: ToolContext,
) -> Dict[str, object]:
    llm_api = _require_llm(context)
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


def writing_tool_functions() -> Dict[str, Callable[..., object]]:
    return {
        "text.summarize": summarize_text,
        "text.extract": extract_corresponding,
        "article.outline": create_outline,
        "article.polish": polish_article,
        "article.generate": generate_article,
    }
