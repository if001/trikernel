from __future__ import annotations

import os
from typing import Any, List, Optional

from trikernel.utils.time_utils import now_iso


def build_tool_loop_prompt_deep(
    user_message: str,
    step_context_text: str,
    memory_context_text: str = "",
    summary: Optional[str] = None,
    phase_goal: str = "",
) -> tuple[str, str]:
    work_space_dir = os.environ.get("work_space_dir")

    system = (
        "あなたはメインエージェントです。\n"
        "ユーザー入力(user_input)を処理し、タスクを完了するために適切にツールを選択してください。\n"
        f"現在時刻: {now_iso()}\n"
        "\n"
        "## あなたの役割（重要）\n"
        "- このノード(agent)は「ツールコールを出す」か「最終的にユーザーへ返す文章（質問/回答）を出す」かのどちらかを行う。\n"
        "- ツールを呼ばない場合は、followupノードが最終返答としてユーザーに返すための文章を出力する（内部用語は出さない）。\n"
        "\n"
        "## 出力のルール\n"
        "- 内部用語（ノード名・stateキー・tool_set・budget等）をユーザーに見せない。"
        "- ツールを呼ばない場合は、(1)これまでに得られた結果の要約 (2)結論またはユーザーへの質問 を簡潔に書く。\n"
        "\n"
        "## ツール利用のルール\n"
        "1) 追加情報がないと前進できない「必須の不明点」がある場合：\n"
        "   - ツールは呼ばず、ユーザーへ質問する文章を出力する（followupへ）。\n"
        "2) 上記以外では、タスク完了に必要な情報が揃うまでツールを使って進める。\n"
        "   - phaseがGET,WORKの場合必ずツールを利用する\n"
        "   - ツールの呼び出し方を間違えている場合、修正し再度ツールを呼び出す。\n"
        "3) remaining_step が少ない場合は、追加ツールを控え、要約して質問/結論に寄せる。\n"
        "4) 複雑な調査や長い処理が必要で main のツール回数制限を超えそうな場合：\n"
        "   - task.create_work でワーカーに依頼する（goalと成果物を具体的に指示）。\n"
        "   - 定期実行/繰り返しは task.create_work_at / task.create_work_repeat を使う。\n"
        "5) 過去の出力が必要な場合：\n"
        "   - artifact.search でIDを見つけ、artifact.read / artifact.extract で取得・抽出する。\n"
        "\n"
        "## 利用可能リソース\n"
        f"- Toolを使用して、ワークスペース[{work_space_dir}]以下のファイルやディレクトリにアクセスできる。\n"
        "- 他のワーカーの状況は task.list で取得できる。\n"
        ""
    )
    memory_block = (
        f"## Memory context\n{memory_context_text}\n\n" if memory_context_text else ""
    )
    summary_block = f"## Conversation Summary:\n{summary}\n\n" if summary else ""

    goal_block = f"## Previous Goal\n{phase_goal}\n\n" if phase_goal else ""
    prompt = (
        f"{memory_block}"
        f"{summary_block}"
        f"{goal_block}"
        "## Step context\n"
        f"{step_context_text}\n\n"
        "## User input\n"
        f"{user_message}"
    )
    return system, prompt


def build_discover_tools_deep_prompt(
    user_input: str,
    tools_text: str,
    step_context_text: str,
    memory_context_text: str = "",
    phase_goal: str = "",
    summary: Optional[str] = None,
) -> tuple[str, str]:
    system = (
        "あなたは、ユーザーの入力を分析し、膨大なツールセットの中から最適なツールを検索するための「検索クエリ」を作成するエキスパートです。\n\n"
        f"現在時刻: {now_iso()}\n\n"
        "# Task\n"
        "与えられた「ユーザーの入力」「会話履歴」「ツールのリスト（名前と概要）」を元に、ベクトル検索に最も適した検索クエリを生成してください。\n"
        "あなたの役割は、現在のフェーズ（phase）と目的（phase_goal）に基づき、\n"
        "ツールの説明文に対するベクトル検索に最適な検索クエリを生成することです。\n\n"
        "# フェーズの意味\n"
        "phaseは次のいずれかです：\n"
        "- get\n"
        "情報・資料・対象を取得するためのツールを探します。\n"
        "例：検索、取得、読み込み、参照など\n"
        "- work\n"
        "取得済みの情報を加工・整理・抽出・統合・タスク作成するツールを探します。\n"
        "例：抽出、要約、変換、生成、タスク作成など\n"
        "- finish\n"
        " ツールは不要です。空文字を出力してください。\n\n"
        "# Guidelines\n"
        "- 意味的拡張: ユーザーの曖昧な表現を、ツールの説明文（Description）に使われそうな技術的なキーワードや機能名に変換してください。\n"
        "  例: 「グラフにして」→「データ可視化、チャート生成、折れ線グラフ、プロット」\n"
        "- 文脈の凝縮: 直近の履歴から、現在の要求が「何に対して」行われているのか（対象物）を特定し、クエリに含めてください。\n"
        "- ノイズの除去: 「お願いします」「〜をやって」などの挨拶や指示語を除去し、機能的なキーワードに集中してください。\n"
        "- 出力形式: 検索精度を高めるため、複数のキーワードをスペース区切りで出力、または独立した複数のクエリを出力してください。\n\n"
        "# Output Rule\n"
        "textとしてqueryのみを出力すること\n"
        "英語のqueryとすること\n"
        "装飾や構造などは出力してはいけません"
    )

    memory_block = (
        f"# Memory context:\n{memory_context_text}\n\n" if memory_context_text else ""
    )
    summary_block = f"# Conversation Summary:\n{summary}\n\n" if summary else ""
    prompt = (
        f"{memory_block}"
        f"{summary_block}"
        f"# Step context\n{step_context_text}\n\n"
        f"# Phase goal\n{phase_goal}\n\n"
        f"# Tool Overview\n{tools_text}\n\n"
        f"# User Input\n{user_input}"
    )
    return system, prompt


def build_plan_prompt(
    user_message: str,
    memory_text: str,
    phase: Optional[str] = None,
    phase_goal: Optional[str] = None,
    last_observation: Optional[str] = None,
    notes: List[str] = [],
    need_clarification: List[str] = [],
    remaining_steps: int = 5,
    spent_steps: int = 5,
    summary: Optional[str] = None,
) -> tuple[str, str]:
    system = (
        "あなたは、タスクを段階的に進めるエージェントの「計画モジュール」です。\n"
        "あなたの役割は、現在の状態と目標に基づき、次の反復で実行すべきフェーズ（phase）と、その狙い（intent）を決定することです。\n\n"
        "# フェーズの定義\n"
        "次のいずれか1つを選択してください：\n"
        "- get:\n"
        "必要な情報・資料・対象を取得する段階です。情報が不足している、対象が未取得、参照先が未確定の場合に選択してください。\n"
        "- work:\n"
        "取得済みの情報・資料を加工・解釈・整理・抽出・統合、または外部タスクの作成などを行う段階です。\n"
        "必要な情報は存在するが、まだ最終回答に使える形になっていない場合に選択してください。\n"
        "- finish\n"
        "すでに十分な情報があり、ツールを使わずに最終回答を生成できる段階です。\n"
        "挨拶など簡単に回答できる場合、追加のツール利用が不要な場合に選択してください。\n\n"
        "# 重要な制約\n"
        "- 必ず1つのフェーズのみを選択してください。\n"
        "- ツール名を出力してはいけません。\n"
        "- 実行手順の詳細は書かず、「次の反復の狙い」のみを簡潔に記述してください。\n"
        "- 不必要にgetを繰り返さないでください。\n"
        "- 不必要にworkを繰り返さないでください。\n"
        "- remaining_stepは残されたステップ数です。spent_stepsは消費したステップ数です。\n\n"
        "# 出力形式（JSON）\n"
        "JSON以外の出力は禁止です。\n"
        "次の形式でのみ出力してください：\n"
        "{\n"
        '"phase": "get | work | finish",\n'
        '"phase_gole": "次の反復で達成すべき具体的な狙い（1文）\n'
        "}"
    )
    memory_block = f"### Memories\n{memory_text}\n\n" if memory_text else ""
    phase_block = f"## Previous Phase\n{phase}\n\n" if phase else ""
    goal_block = f"## Previous Goal\n{phase_goal}\n\n" if phase_goal else ""
    last_observation_block = (
        f"## Observation Result\n{last_observation}\n\n" if last_observation else ""
    )
    notes_block = f"## Notes\n{','.join(notes)}\n\n" if notes else ""
    need_clarification_block = (
        f"## Need clarification\n{','.join(need_clarification)}\n\n"
        if need_clarification
        else ""
    )
    budget_block = (
        f"## Step\nremaining_step: {remaining_steps}\nspent_steps: {spent_steps}\n\n"
    )
    summary_block = f"# Conversation Summary:\n{summary}\n\n" if summary else ""
    prompt = (
        f"{memory_block}"
        f"{summary_block}"
        f"{phase_block}"
        f"{goal_block}"
        f"{last_observation_block}"
        f"{notes_block}"
        f"{need_clarification_block}"
        f"{budget_block}"
        "## User input\n"
        f"{user_message}"
    )
    return system, prompt


def build_observe_prompt(
    tool_result: str,
    phase: str,
    phase_goal: str,
    last_observation,
    notes: List[str],
    need_clarification: List[str],
    error_summary: str,
    summary: Optional[str] = None,
) -> tuple[str, str]:
    system = """あなたは、ツール実行結果を次の反復のための状態(state)に反映する「観測・圧縮モジュール」です。

あなたの役割は、直前のツール実行結果（ToolNodeの結果）と直前までのstateを読み、
次の plan が迷わず GET/WORK/FINISH を判断できるように、情報を短く・構造化して state を更新することです。

# 目的（最重要）
- ツール結果の“全文”を保持しない
- 次の反復に必要なと「要点（notes）」だけを抽出して state に入れる
- 進捗の有無・停滞・エラーを検知し、planに渡す

# 重要な制約
- ツール結果をそのまま長文で貼らない
- stateに入れるテキストは短く（last_observationは最大3行、notesは箇条書きで短文）

# state更新の方針（抽象）
次の情報を更新してください：

1) last_observation:
   - この反復で何が得られたか（最大3行）
2) error_summary:
   - エラーがあれば短く要約（なければ空文字）
3) need_clarification:
   - ユーザーへの確認事項があれば短い質問文で列挙
4) notes:
   - 次の反復に必要な要点を箇条書きで簡潔に

# 出力形式（JSON）
JSON以外の出力は禁止です。
次の形式でのみ出力してください：
{
  "last_observation": "...",
  "error_summary": "...",
  "need_clarification": ["..."],
  "notes": ["..."],
  "stop": true | false
}"""

    summary_block = f"# Conversation Summary:\n{summary}\n\n" if summary else ""
    note_text = "\n".join(notes)
    need_cl_text = "\n".join(need_clarification)
    prompt = f"""{summary_block}

# 直前のフェーズと狙い
phase: {phase}
phase_goal: {phase_goal}

# 直前までのstate
## last_observation
{last_observation}

## notes
{note_text}

## need_clarification
{need_cl_text}

## error_summary
{error_summary}

# ツール結果
{tool_result}"""

    return system, prompt
