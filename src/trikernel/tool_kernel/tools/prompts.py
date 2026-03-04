from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def build_step_goal_prompt(
    previous_goal: Optional[str],
    failure_reason: Optional[str],
    step_context: Optional[Dict[str, Any]],
    user_message: Optional[str],
    task_payload: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> str:
    return (
        "You are deciding the current step goal for a task.\n"
        "If a previous_goal exists, keep it unless there is a clear reason to adjust it.\n"
        "If the previous step failed, incorporate the failure_reason or open issues to refine the goal.\n"
        "Use the latest user_message or worker input when it is more specific than previous_goal.\n"
        "Respond in JSON with keys `step_goal` and `rationale`.\n"
        f"Previous goal: {previous_goal}\n"
        f"Failure reason: {failure_reason}\n"
        f"Step context: {json.dumps(step_context or {}, ensure_ascii=False)}\n"
        f"User/worker input: {user_message}\n"
        f"Task payload: {json.dumps(task_payload, ensure_ascii=False)}\n"
        f"Recent turns: {json.dumps(history, ensure_ascii=False)}"
    )


def build_summary_prompt(
    text: str,
    max_length: Optional[int],
    style: Optional[str],
    language: Optional[str],
) -> str:
    return (
        "Summarize the input text clearly and concisely.\n"
        "Return JSON with keys `summary`, `bullets` (array), and `notes`.\n"
        f"Max length: {max_length}\n"
        f"Style: {style}\n"
        f"Language: {language}\n"
        f"Text: {text}"
    )


def build_extract_prompt(
    source_text: str,
    target_text: str,
    criteria: Optional[str],
    language: Optional[str],
) -> str:
    return (
        "Extract the parts of target_text that correspond to the content in source_text.\n"
        "Be selective and preserve original wording from target_text.\n"
        "Return JSON with keys `extracted` (array of strings) and `notes`.\n"
        f"Criteria: {criteria}\n"
        f"Language: {language}\n"
        f"Source text: {source_text}\n"
        f"Target text: {target_text}"
    )


def build_outline_prompt(
    user_input: Optional[str],
    tool_results: Optional[List[str]],
    article_type: Optional[str],
    audience: Optional[str],
    language: Optional[str],
) -> str:
    return (
        "Create a structured article outline based on the inputs.\n"
        "Return JSON with keys `title`, `sections` (array of {heading, bullets}), and `notes`.\n"
        f"Article type: {article_type}\n"
        f"Audience: {audience}\n"
        f"Language: {language}\n"
        f"User input: {user_input}\n"
        f"Tool results: {json.dumps(tool_results or [], ensure_ascii=False)}"
    )


def build_polish_prompt(
    draft: str,
    article_type: Optional[str],
    audience: Optional[str],
    language: Optional[str],
) -> str:
    return (
        "Polish the draft from an editor's perspective.\n"
        "Fix clarity, structure, and tone while preserving meaning.\n"
        "Return JSON with keys `revised`, `edits` (array of strings), and `notes`.\n"
        f"Article type: {article_type}\n"
        f"Audience: {audience}\n"
        f"Language: {language}\n"
        f"Draft: {draft}"
    )


def build_article_prompt(
    article_type: str,
    audience: str,
    revisions: Optional[List[str]],
    outline: Optional[str],
    draft: str,
    language: Optional[str],
) -> str:
    return (
        "Write a complete article from the draft and outline.\n"
        "Apply the revision points. Keep the structure clear and consistent.\n"
        "Return JSON with keys `article` and `notes`.\n"
        f"Article type: {article_type}\n"
        f"Audience: {audience}\n"
        f"Language: {language}\n"
        f"Revisions: {json.dumps(revisions or [], ensure_ascii=False)}\n"
        f"Outline: {outline}\n"
        f"Draft: {draft}"
    )
