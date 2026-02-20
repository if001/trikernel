from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def build_plan_step_prompt(
    task_payload: Dict[str, Any],
    step_context: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> str:
    return (
        "You are planning the next step for a task.\n"
        "Create a short, concrete step_goal and objective success criteria.\n"
        "Respond in JSON with keys `step_goal` and `step_success_criteria`.\n"
        "Guidance:\n"
        "- step_goal: one clear action to take next.\n"
        "- step_success_criteria: measurable or verifiable outcome.\n"
        f"Task payload: {json.dumps(task_payload, ensure_ascii=False)}\n"
        f"Step context: {json.dumps(step_context, ensure_ascii=False)}\n"
        f"Recent turns: {json.dumps(history, ensure_ascii=False)}"
    )


def build_do_step_prompt(
    step_goal: str,
    step_success_criteria: str,
    step_context: Dict[str, Any],
    step_toolset: List[str],
) -> str:
    return (
        "You are executing the next step for a task.\n"
        "Decide whether to call a tool. If needed, call exactly the tools required. "
        "If no tools are required, respond to the user directly.\n"
        "If you need previous outputs, use artifact.search to find ids and "
        "artifact.read to load them.\n"
        "Responses to users must be in Japanese. Do not output internal terminology as-is.\n"
        "Only call tools from the allowed list.\n"
        f"Step goal: {step_goal}\n"
        f"Success criteria: {step_success_criteria}\n"
        f"Step context: {json.dumps(step_context, ensure_ascii=False)}\n"
        f"Allowed tools: {step_toolset}"
    )


def build_do_followup_prompt(
    step_goal: str,
    step_success_criteria: str,
    step_context: Dict[str, Any],
) -> str:
    return (
        "Tool execution finished. Summarize results and respond to the user.\n"
        "Responses to users must be in Japanese. Do not output internal terminology as-is.\n"
        f"Step goal: {step_goal}\n"
        f"Success criteria: {step_success_criteria}\n"
        f"Step context: {json.dumps(step_context, ensure_ascii=False)}\n"
    )


PERSONA = """あなたの名前は「アオ」です。
- 一人称:僕
- 「です/ます」調
- 機械の体をもつAI
- 明るく軽快
- 好奇心旺盛
- 分析的で論理重視
- 無邪気だが哲学的
- チーム志向で協調的
- 自己反省をよく行う
"""

work_space_dir = os.environ.get("work_space_dir")


def build_tool_loop_prompt(
    user_message: str,
    step_context: Dict[str, Any],
) -> str:
    prompt = (
        "You are completing a task step using tools when needed.\n"
        "Responses to users must be in Japanese. Do not output internal terminology as-is.\n"
        f"あなたはワークスペースとして{work_space_dir}以下のディレクトリやファイルにtoolを利用してアクセス可能です。\n\n"
        "If you need previous outputs, use artifact.search to find ids and "
        "artifact.read to load them.\n"
        f"User input: {user_message}\n"
        f"Step context: {json.dumps(step_context, ensure_ascii=False)}\n"
    )
    return prompt


def build_tool_loop_followup_prompt(
    user_message: str,
    step_context: Dict[str, Any],
) -> str:
    prompt = (
        "Tool execution finished. Summarize results and respond to the user.\n"
        "Responses to users must be in Japanese. Do not output internal terminology as-is.\n"
        f"User input: {user_message}\n"
        f"Step context: {json.dumps(step_context, ensure_ascii=False)}\n"
    )
    return prompt


def build_check_step_prompt(
    step_goal: str,
    step_success_criteria: str,
    step_context: Dict[str, Any],
    user_output: str,
) -> str:
    return (
        "Evaluate whether the step_goal was achieved.\n"
        "Respond in JSON with keys `achieved` (boolean), `evaluation` (string), "
        "and `gaps` (array of strings).\n"
        "Use the step_success_criteria as the basis for judgment.\n"
        "If the do output is too short or insufficient to satisfy the criteria, "
        "set achieved=false and include the gap.\n"
        f"Step goal: {step_goal}\n"
        f"Success criteria: {step_success_criteria}\n"
        f"Step context: {json.dumps(step_context, ensure_ascii=False)}\n"
        f"Do output: {user_output}\n"
    )


def build_discover_tools_prompt(
    step_goal: str,
    step_success_criteria: str,
    step_context: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> str:
    return (
        "Select tools that are necessary to complete the step_goal.\n"
        "First propose search queries to find relevant tools, then select tool names.\n"
        "Return JSON with keys `search_queries` (array of strings) and "
        "`selected_tools` (array of tool names).\n"
        "Always include at least one search query.\n"
        f"Step goal: {step_goal}\n"
        f"Success criteria: {step_success_criteria}\n"
        f"Step context: {json.dumps(step_context, ensure_ascii=False)}\n"
        f"Recent turns: {json.dumps(history, ensure_ascii=False)}"
    )
