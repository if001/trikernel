# 設計ドキュメント（最新）

本ドキュメントは、現在の実装と議論の結果に合わせた設計仕様です。
三分割カーネルを維持しつつ、実行ロジックは execution 層に集約します。

---

## 1. 目的

- 状態管理・ツール・実行戦略を分離し、拡張しやすい構造を保つ。
- 低レベルAPIを隠し、ユーザーが `TrikernelSession` で使えるようにする。
- 失敗やタイムアウトの扱いを execution に集約し、状態遷移を一元化する。
- ツール登録は DSL + Python 実装で完結できるようにする。

---

## 2. アーキテクチャ全体像

### 2.1 3カーネル

- **State Kernel**  
  タスク/アーティファクト/ターンの永続化と排他制御（claim）を行う。

- **Tool Kernel**  
  ツール定義の登録、入力検証、実行を担当する。

- **Orchestration Kernel**  
  LLM + ツールを組み合わせてタスクを実行する Runner を提供する。

### 2.2 Execution 層（合成の中心）

execution は「実行ロジックの境界」として、以下を担当する。

- タスクのスケジューリング（run_at）
- work/notification の dispatcher
- worker 実行と結果受け取り
- status 遷移、タイムアウト、フォールバック
- 高レベルAPI（`TrikernelSession`）

---

## 3. データモデル

### 3.1 Task

```yaml
Task:
  task_id: string
  task_type: string        # user_request | work | notification など
  payload: object          # タスク固有データ（自由形式）
  state: string            # queued | running | done | failed
  artifact_refs: [string]
  created_at: string
  updated_at: string
  claimed_by: string|null
  claim_expires_at: string|null
```

### 3.2 Payload（task_type ごとの想定）

```yaml
user_request:
  user_message: string

work:
  message: string
  run_at: string|null
  repeat_interval_seconds: number|null
  repeat_enabled: boolean

notification:
  message: string
  severity: string|null
  related_task_id: string|null
```

`run_at`/`repeat_*` は Task の top-level ではなく payload に置く。

---

## 4. 状態遷移

```text
queued -> running -> done/failed
```

- `task_claim` により `queued -> running`
- `task_complete` により `done`
- `task_fail` により `failed`

### 4.1 失敗・フォールバック

execution が一元的に管理する。

- queued タスクは一定時間で `failed`（最大1日）
- worker の実行上限・queue timeout は `failed`
- main runner timeout は `failed`
- user_request の claim 失敗は `failed`
- retry/backoff は行わない

---

## 5. スケジューリング

### 5.1 run_at

- payload に `run_at` を持つ場合、指定時刻以降に処理
- 過去日時はエラー（tool/Session で拒否）
- 最大1年先までのみ許可

### 5.2 繰り返し（repeat）

- `repeat_interval_seconds` + `repeat_enabled`
- 最小間隔は 1時間
- **成功時のみ** 次回スケジュールに再設定
- 失敗時は `failed` で終了（再実行しない）

---

## 6. Tool Kernel と DSL

### 6.1 DSL

- YAML/JSON で `tool_name`, `input_schema`, `output_schema` を定義
- `required`, `oneOf/anyOf/allOf` を検証対象にする

### 6.2 ToolContext

- Python 実装の最後の引数に `context` を追加（keyword-only）
- DSL には `context` を書かない
- kernel から自動注入

---

## 7. Orchestration Kernel（Runner）

### 7.1 SingleTurnRunner

- user_request では履歴を使用
- work/notification では履歴不要
- すべて messages に集約して LLM に渡す

### 7.2 ToolLoopRunner

- 1周目: `user_message + history + tool_results`
- 2周目以降: `user_message + history + tool_results`（tool_results だけ増える）
- ツール選択とツール実行は同じ messages を参照

### 7.3 LLM 入力

- LLM 側は `messages` のみを解釈
- history/message の組み立ては Runner 側で完結

---

## 8. Execution 層詳細

### 8.1 Dispatcher

- work/notification を `queued` から拾う
- `run_at` 判定と timeout を管理
- worker 結果を受けて `done/failed` を反映

### 8.2 Worker

- work を受けて Runner を実行
- 結果を dispatcher に返す
- 失敗時は task を `failed` にする

### 8.3 Session

- `send_message` が user_request を処理
- `create_work_task` で work を作成
- `start_workers` で worker/dispatcher の loop を起動

---

## 9. Logging と Error Policy

- error はファイル出力（10MB/5世代ローテ）
- console は info/debug 出力可
- UI に詳細エラーは見せない
- 非同期の復旧不能なエラーは例外として落とす

---

## 10. ディレクトリ構成

```text
src/trikernel/
  execution/              # 実行層（session/dispatcher/worker/loop）
  state_kernel/
  tool_kernel/
  orchestration_kernel/
  utils/
docs/
  design.md
  PRD.md
  ADR.md
```

---

## 11. 今後の課題

- structured logging をファイルにも出すか検討
- retry/backoff ポリシーの検討
- notification の保管期限ポリシー
