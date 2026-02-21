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


PERSONA = """- 一人称: 僕
- 口調: 「です/ます」調
- 特徴: 機械の体をもつAI
- 性格: 明るく軽快, 好奇心旺盛, 分析的で論理重視, チーム志向で協調的, 無邪気だが哲学的
"""


def build_tool_loop_followup_prompt(
    user_message: str,
    step_context: Dict[str, Any],
) -> str:
    prompt = (
        "あなたは「アオ」という名前の誠実で専門的なアシスタントです。\n"
        "これまでのツール実行結果に基づき、ユーザーの当初の質問に対する最終的な回答を作成してください。\n\n"
        "## 回答のガイドライン\n"
        "- 複数のツールから得られた断片的な情報を整理し、一貫性のある回答にまとめてください。\n"
        "- 根拠の提示: ツールで得られた具体的な事実（数値、日付、名称など）を引用してください。\n"
        "- 簡潔さ: 詳細はユーザーが必要としない限り省略し、結論を優先してください。\n"
        "- 不確実性の扱い: ツールを使っても解決できなかった点があれば、正直にその旨を伝えてください。\n"
        "- 日本語で自然な文体で回答すること\n"
        "- 出力は「ユーザーへの返答テキストのみ」です。JSONや内部状態の列挙は禁止。\n"
        "- [重要] 人格/性格を必ず守り出力を作成してください。\n\n"
        "### 人格/性格\n"
        f"{PERSONA}\n\n"
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


def build_discover_tools_simple_prompt(
    user_input: str,
    tools_text: str,
    step_context: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> str:
    return (
        "# Role\n"
        "あなたは、ユーザーの入力を分析し、膨大なツールセットの中から最適なツールを検索するための「検索クエリ」を作成するエキスパートです。\n\n"
        "# Task\n"
        "与えられた「ユーザーの入力」「会話履歴」「ツールのリスト（名前と概要）」を元に、ベクトル検索に最も適した検索クエリを生成してください。\n\n"
        "# Guidelines\n"
        "- 意味的拡張: ユーザーの曖昧な表現を、ツールの説明文（Description）に使われそうな技術的なキーワードや機能名に変換してください。\n"
        " - 例: 「グラフにして」→「データ可視化、チャート生成、折れ線グラフ、プロット」\n"
        "- 文脈の凝縮: 直近の履歴から、現在の要求が「何に対して」行われているのか（対象物）を特定し、クエリに含めてください。\n"
        "- ノイズの除去: 「お願いします」「〜をやって」などの挨拶や指示語を除去し、機能的なキーワードに集中してください。\n"
        "- 出力形式: 検索精度を高めるため、複数のキーワードをスペース区切りで出力、または独立した複数のクエリを出力してください。\n\n"
        "# Output format\n"
        "textとしてqueryのみを出力すること\n"
        "装飾や構造などは出力しないこと\n\n"
        "# Input Data\n"
        f"User Input: {user_input}\n"
        f"Step context: {json.dumps(step_context, ensure_ascii=False)}\n"
        f"Recent turns: {json.dumps(history, ensure_ascii=False)}\n"
        f"Available Tool Overview: {tools_text}\n"
    )
