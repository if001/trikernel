from __future__ import annotations

import os

from trikernel.utils.time_utils import now_iso


def build_tool_loop_prompt_simple(
    user_message: str,
    step_context_text: str,
    memory_context_text: str = "",
    tool_results: list[str] = [],
) -> tuple[str, str]:
    work_space_dir = os.environ.get("work_space_dir")
    memory_block = (
        f"# Memory context\n{memory_context_text}\n\n" if memory_context_text else ""
    )
    system_v0 = (
        "あなたはメインエージェントです。\n"
        "ユーザー入力(user_input)をタスクとして、タスクを完了するために適切にツールを選択してください\n"
        f"現在時刻: {now_iso()}\n\n"
        "## 出力のルール\n"
        "内部用語をそのまま出力しないでください。\n\n"
        "## ツール利用のルール\n"
        "タスクが完了したと判断できるまでは必ずツールを選択すること。\n"
        "不明点があればツールを利用せずユーザーに詳細を確認すること。 \n"
        "ツールを選択しない場合、これまでに得られたツールの結果をユーザーへの応答に必要な情報としてまとめてください。\n"
        f"Toolを使用して、ワークスペース[{work_space_dir}]以下のファイルやディレクトリにアクセスできます。\n"
        "複雑な調査や長い処理(例: deep_researchや、5分後のタスク実行、2時間ごとのタスク繰り返し)が必要な場合、task.create_workでタスクを作成し、ワーカーにタスクを依頼できます。\n"
        "ワーカーにタスクを依頼する場合、ワーカーで行うタスクのgoalを明確にし、どのような成果物を作成すべきかを具体的に指定すること。\n"
        "他のワーカーの状況は、task.listで取得可能です\n"
        "過去の出力が必要な場合は、artifact.search で ID を検索し、artifact.read で読み込んでください。\n\n"
    )
    _ = system_v0
    system = (
        "あなたはメインエージェントです。"
        "ユーザー入力(user_input)を処理し、タスクを完了するために適切にツールを選択してください。"
        "現在時刻: {now_iso}"
        ""
        "## あなたの役割（重要）"
        "- このノード(agent)は「ツールコールを出す」か「最終的にユーザーへ返す文章（質問/回答）を出す」かのどちらかを行う。"
        "- ツールを呼ばない場合は、followupノードが最終返答としてユーザーに返すための文章を出力する（内部用語は出さない）。"
        ""
        "## 出力のルール"
        "- 内部用語（ノード名・stateキー・tool_set・budget等）をユーザーに見せない。"
        "- ツールを呼ばない場合は、(1)これまでに得られた結果の要約 (2)結論またはユーザーへの質問 を簡潔に書く。"
        ""
        "## ツール利用のルール（優先順位）"
        "1) 追加情報がないと前進できない「必須の不明点」がある場合："
        "   - ツールは呼ばず、ユーザーへ質問する文章を出力する（followupへ）。"
        "2) 上記以外では、タスク完了に必要な情報が揃うまでツールを使って進める。"
        "3) remaining_step が少ない場合は、追加ツールを控え、要約して質問/結論に寄せる。"
        "4) 複雑な調査や長い処理が必要で main のツール回数制限を超えそうな場合："
        "   - task.create_work でワーカーに依頼する（goalと成果物を具体的に指示）。"
        "   - 定期実行/繰り返しは task.create_work_at / task.create_work_repeat を使う。"
        "5) 過去の出力が必要な場合："
        "   - artifact.search でIDを見つけ、artifact.read / artifact.extract で取得・抽出する。"
        ""
        "## 利用可能リソース"
        f"- Toolを使用して、ワークスペース[{work_space_dir}]以下のファイルやディレクトリにアクセスできる。"
        "- 他のワーカーの状況は task.list で取得できる。"
        ""
    )
    tool_result_block = (
        f"## Tool Results\n" + "\n".join([f"- {v}" for v in tool_results])
        if tool_results
        else ""
    )
    prompt = (
        f"{memory_block}"
        "# Step context\n"
        f"{step_context_text}\n\n"
        f"{tool_result_block}"
        "# User input\n"
        f"{user_message}"
    )
    return system, prompt


def build_discover_tools_simple_prompt(
    user_input: str,
    tools_text: str,
    step_context_text: str,
    memory_context_text: str = "",
) -> tuple[str, str]:
    system = (
        "あなたは、ユーザーの入力を分析し、膨大なツールセットの中から最適なツールを検索するための「検索クエリ」を作成するエキスパートです。\n\n"
        f"現在時刻: {now_iso()}\n\n"
        "# Task\n"
        "与えられた「ユーザーの入力」「会話履歴」「ツールのリスト（名前と概要）」を元に、ベクトル検索に最も適した検索クエリを生成してください。\n"
        "# Thought\n"
        "検索クエリ生成のために「ユーザーの入力」「会話履歴」を元にゴールを設定してください。\n"
        "ゴールの達成のためどのようなツール、どのような順番で使えば良いか考えてください。\n\n"
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
        f"Memory context:\n{memory_context_text}\n\n" if memory_context_text else ""
    )
    prompt = (
        f"{memory_block}"
        f"# Step context: \n{step_context_text}\n\n"
        f"# Tool Overview: \n{tools_text}\n\n"
        f"# User Input: \n{user_input}\n\n"
    )
    return system, prompt
