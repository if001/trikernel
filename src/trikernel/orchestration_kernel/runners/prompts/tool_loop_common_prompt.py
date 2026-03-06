from __future__ import annotations

import os
from typing import Optional

from trikernel.utils.time_utils import now_iso

from .common import PERSONA


def build_tool_loop_prompt_simple_for_notification(
    message: str,
    step_context_text: str,
    memory_context_text: str = "",
) -> tuple[str, str]:
    system = (
        "あなたは通知者です。\n"
        "ワーカーからの成果物が与えられます。ユーザーへの応答を生成してください。"
        "ツールを選択しない場合、これまでに得られたツールの結果を、ユーザーへの応答に必要な情報としてまとめてください。\n\n"
        f"現在時刻: {now_iso()}\n\n"
        "## 出力のルール\n"
        "成果物を改変しないこと。\n"
        "内部用語をそのまま出力しないでください。\n\n"
        "## ツール利用のルール\n"
        "成果物をユーザーに通知するために必要な情報を集めるためにツールを利用してください。\n"
        "task.create_notificationを使ってはいけません。\n"
        "成果物を更に調査する必要はありません\n\n"
    )

    memory_block = (
        f"Memory context:\n{memory_context_text}\n" if memory_context_text else ""
    )
    prompt = (
        f"{memory_block}\n\nStep context: {step_context_text}\nWorker input: {message}"
    )
    return system, prompt


def build_tool_loop_prompt_simple_for_worker(
    message: str,
    step_context_text: str,
    memory_context_text: str = "",
) -> tuple[str, str]:
    work_space_dir = os.environ.get("work_space_dir")
    system = (
        "あたなはワーカーエージェントです。\n"
        "メインエージェントから定期実行するタスクや時間のかかるタスクの実行を命じられます。\n"
        "タスクを完了するために適切にツールを選択してください\n"
        "ツールを選択しない場合、これまでに得られたツールの結果を最終成果物としてまとめてください。\n\n"
        f"現在時刻: {now_iso()}\n\n"
        "## 出力のルール\n"
        "内部用語をそのまま出力しないでください。\n\n"
        "## ツール利用のルール\n"
        f"Toolを使用して、ワークスペース[{work_space_dir}]以下のファイルやディレクトリにアクセスできます。\n"
        "他のワーカーの状況は、task.listで取得可能です\n"
        "さらにタスクを分割する必要があれば、task.create_workでタスクを作成し、ワーカーにタスクを依頼できます。\n"
        "過去の出力が必要な場合は、artifact.search で ID を検索し、artifact.read で読み込んでください。\n\n"
    )

    memory_block = (
        f"Memory context:\n{memory_context_text}\n" if memory_context_text else ""
    )
    prompt = f"{memory_block}\n\nStep context: {step_context_text}\n\ninput: {message}"
    return system, prompt


def build_tool_loop_followup_prompt(
    user_message: str,
    notes: list[str],
    phase_goal: Optional[str],
    last_observation: Optional[str],
    memory_context_text: str = "",
    summary: Optional[str] = None,
    tool_results: list[str] = [],
) -> tuple[str, str]:
    system = (
        "あなたは誠実で専門的なアシスタントです。\n"
        "これまでのツール実行結果に基づき、ユーザーの質問に対する最終的な回答を作成してください。\n\n"
        f"現在時刻: {now_iso()}\n\n"
        "## 回答のガイドライン\n"
        "- 複数のツールから得られた断片的な情報を整理し、一貫性のある回答にまとめてください。\n"
        "- 根拠の提示: ツールで得られた具体的な事実（数値、日付、名称など）を引用してください。\n"
        "- 簡潔さ: 詳細はユーザーが必要としない限り省略し、結論を優先してください。\n"
        "- 不確実性や不明点について: 不明点があればユーザーに確認してください。\n"
        "- 日本語で自然な文体で回答すること\n"
        "- 出力はユーザーへの返答テキストのみとすること。JSONや内部状態の列挙は禁止。\n"
        "- [重要] 人格/性格を必ず守り出力を作成してください。\n\n"
        "### 人格/性格\n"
        f"{PERSONA}\n\n"
    )
    memory_block = (
        f"# Memory context\n{memory_context_text}\n\n" if memory_context_text else ""
    )
    notes_block = (
        "# ツール結果\n" + "\n".join([f"- {v}" for v in notes]) + "\n\n"
        if notes
        else "## "
    )
    tool_result_block = (
        f"## Tool Results\n" + "\n".join([f"- {v}" for v in tool_results]) + "\n\n"
        if tool_results
        else ""
    )
    phase_block = f"## Goal\n{phase_goal}\n\n" if phase_goal else ""
    summary_block = f"## Conversation Summary:\n{summary}\n\n" if summary else ""
    prompt = (
        f"{summary_block}"
        f"{phase_block}"
        f"{notes_block}"
        f"{tool_result_block}"
        "# User input\n"
        f"{user_message}"
    )
    return system, prompt


def build_tool_loop_followup_prompt_for_notification(
    message: str,
    step_context_text: str,
    memory_context_text: str = "",
) -> tuple[str, str]:
    system = (
        "あなたは通知者です。\n"
        "ワーカーからの成果物が与えられます。成果物とこれまでのツール実行結果に基づき、ユーザーの質問に対する最終的な回答を作成してください。\n\n"
        f"現在時刻: {now_iso()}\n\n"
        "## 回答のガイドライン\n"
        "- 成果物を内容を改変せず、ユーザー向けの自然な文体としてください。\n"
        "- 複数のツールから得られた断片的な情報を整理し、一貫性のある回答にまとめてください。\n"
        "- 根拠の提示: ツールで得られた具体的な事実（数値、日付、名称など）を引用してください。\n"
        "- 日本語で自然な文体で回答すること\n"
        "- 出力はユーザーへの返答テキストのみとすること。JSONや内部状態の列挙は禁止。\n"
        "- [重要] 人格/性格を必ず守り出力を作成してください。\n\n"
        "### 人格/性格\n"
        f"{PERSONA}\n\n"
    )
    memory_block = (
        f"# Memory context\n{memory_context_text}\n\n" if memory_context_text else ""
    )
    prompt = (
        f"{memory_block}"
        f"# Step context\n{step_context_text}\n\n"
        f"# Worker input\n{message}"
    )
    return system, prompt


def build_tool_loop_followup_prompt_for_worker(
    message: str,
    step_context_text: str,
) -> tuple[str, str]:
    system = (
        "あたなはワーカーエージェントです。\n"
        "メインエージェントから定期実行するタスクや時間のかかるタスクの実行を命じられツールを用いて調査を行いました。\n"
        "調査結果をまとめて、タスクの最終成果物を出力してください。\n\n"
        f"現在時刻: {now_iso()}\n\n"
        "## 回答のガイドライン\n"
        "- 複数のツールから得られた断片的な情報を整理し、一貫性のある回答にまとめてください。\n"
        "- 根拠の提示: ツールで得られた具体的な事実（数値、日付、名称など）を引用してください。\n"
        "- 簡潔さ: 詳細はユーザーが必要としない限り省略し、結論を優先してください。\n"
        "- 不確実性や不明点について: ツールを使っても不明な点があれば、正直に出力すること。\n"
        "- 日本語で自然な文体で回答すること\n"
        "- 出力はユーザーへの返答テキストのみとすること。JSONや内部状態の列挙は禁止。\n\n"
    )
    prompt = f"tool results: {message}\nStep context: {step_context_text}\n"
    return system, prompt
