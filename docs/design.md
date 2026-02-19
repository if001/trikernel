# docs

本設計は、LLM Agent を **3つの独立したカーネル（核）**に分離し、それらを **合成（組み合わせ）**することで、従来の「PDCA-Manager / PDCA-Run / Step」相当のシステムを構築するための設計仕様である。

* **状態カーネル（State Kernel）**：タスクと生成物（Artifact）を永続化し、排他取得（claim）を提供する状態の正（Single Source of Truth）
* **ツールカーネル（Tool Kernel）**：ツール定義を登録し、入力検証・実行・出力検証を行うツール実行基盤
* **オーケストレーションカーネル（Orchestration Kernel）**：LLM とツール呼び出しを組み合わせてタスクを進める実行戦略（PDCA 等）を提供する実行器

本設計では、**各カーネルは他カーネルの内部実装を知らない**。カーネル間の結合は、明確に定義された **インターフェース（公開API）**を介してのみ行う。

---

## 0. 用語集（本書内での意味を固定）

* **カーネル**：単体で動作可能な中核モジュール。公開APIを持ち、他カーネルの内部実装に依存しない。
* **合成（Composition）**：複数カーネルを接続し、1つのシステムとして動作させること。
* **LLM**：大規模言語モデル。入力テキストから出力テキストを生成し、必要に応じて「ツール呼び出し指示」を出す。
* **ツール（Tool）**：外部機能を呼び出すための関数的インターフェース。入力と出力の形式が定義される。副作用（状態更新や書き込み）が発生してもよい。
* **ツール定義（Tool Definition）**：ツールの名前、説明、入力形式、出力形式、実装参照をまとめたメタ情報。
* **実装参照（Implementation Reference）**：ツール定義から生成される実行可能実装（関数やハンドラ）への参照。ツール定義と実装を分離するために用いる。
* **タスク（Task）**：処理の単位。キューに入り、実行され、完了または失敗になる。
* **生成物（Artifact）**：ツールやLLMが生み出した成果物（テキスト、JSONなど）。ArtifactStoreに保存され、IDで参照される。
* **排他取得（Claim）**：複数実行者が同じタスクを同時に処理しないために、特定タスクを「自分が処理する」と確定する操作。
* **実行者（Runner）**：タスクを受け取り、LLMとツールを用いて処理するコンポーネント。オーケストレーションカーネルの中核。
* **実行戦略（Orchestration Strategy）**：Runnerがタスクをどう進めるかの方針。例：単発実行、PDCA反復実行。
* **Step（ステップ）**：実行戦略が複数回の反復を行う場合の1回分。PDCA戦略では「1 Step = 1 PDCAサイクル」。
* **Step Context（ステップ文脈）**：反復実行の際に次のステップへ持ち越す最小の作業メモ。
* **クライアント（Client）**：ユーザー入力を渡し、結果を受け取る外部（UI、CLIなど）。

---

## 1. 設計目的

1. **分離**：状態管理、ツール実行、実行戦略を独立した核として分ける
2. **合成**：核を組み合わせることで「シンプル版」「PDCA版」「並列版」を構成できる
3. **拡張**：DSLや並列処理、PDCAなどを特定カーネルの追加・差し替えで実現できる
4. **実装容易性**：最初の実装を「核ごとの小さな実装」に分解して進められる

---

## 2. 非目的（本設計で扱わないこと）

* 具体的なLLMプロバイダの選定や課金設計
* UI（画面設計、デザイン）
* ネットワーク境界（HTTP/IPC）の詳細なプロトコル仕様（API形は定義するが通信形態は実装任せ）
* セキュアサンドボックスの完全仕様（最小限の考慮のみ）

---

## 3. 全体構成（3カーネル + 合成層）

### 3.1 カーネル一覧

* 状態カーネル（State Kernel）
* ツールカーネル（Tool Kernel）
* オーケストレーションカーネル（Orchestration Kernel）

### 3.2 合成層（Composition Layer）

合成層は「カーネルを接続する薄い層」であり、以下の責務のみを持つ。

* クライアントからの入力をタスク化して状態カーネルへ登録
* 状態カーネルから処理対象タスクを排他取得（claim）
* オーケストレーションカーネルにタスク処理を依頼
* 返答をクライアントへ返す（返答がある場合）

合成層は、カーネルの内部ロジックを持たない。カーネルが提供するAPI呼び出しの組み立てだけを行う。

---

## 4. 状態カーネル（State Kernel）

### 4.1 責務

状態カーネルはシステムにおける状態の正（Single Source of Truth）である。

* タスクの永続化（TaskStore）
* 生成物の永続化（ArtifactStore）
* タスクの排他取得（claim）と状態遷移の整合性維持
* タスクと生成物の参照関係を保存

状態カーネルは、LLM、PDCA、ツール実行などの概念を持たない。

### 4.2 データモデル

#### 4.2.1 Task

Task は以下の最小フィールドを持つ。

```yaml
Task:
  task_id: string
  task_type: string          # 例: user_request, work, notification
  payload: object            # タスク固有データ（自由形式）
  state: string              # queued | running | done | failed
  artifact_refs:             # 関連する生成物ID
    - string
  created_at: string         # ISO 8601
  updated_at: string         # ISO 8601
  claimed_by: string|null    # claimした実行者識別子
  claim_expires_at: string|null # claimの期限（タイムアウト用）
```

* `task_type` は文字列で、状態カーネルは意味解釈しない。
* `payload` は自由形式だが、オーケストレーション側で解釈する。
* `claimed_by` と `claim_expires_at` は排他制御に使用する。

#### 4.2.2 Artifact

Artifact は以下の最小フィールドを持つ。

```yaml
Artifact:
  artifact_id: string
  media_type: string         # 例: text/markdown, application/json
  body: string               # 実体（当面は文字列に限定してよい）
  metadata: object           # 任意メタ情報（自由形式）
  created_at: string         # ISO 8601
```

### 4.3 状態カーネル公開API（最小）

#### 4.3.1 Task API

* `task_create(task_type, payload) -> task_id`
* `task_get(task_id) -> Task`
* `task_update(task_id, patch) -> Task`
* `task_list(filter) -> Task[]`
* `task_claim(filter, claimer_id, ttl_seconds) -> task_id|null`
* `task_complete(task_id) -> Task`
* `task_fail(task_id, error_info) -> Task`

補足：

* `patch` は部分更新（JSON Merge Patch相当）を想定するが、具体形式は実装任せ。
* `task_claim` は **atomic（原子的）**に行われる必要がある。

#### 4.3.2 Artifact API

* `artifact_write(media_type, body, metadata) -> artifact_id`
* `artifact_read(artifact_id) -> Artifact`
* `artifact_search(query) -> artifact_id[]`

### 4.4 排他制御（Claim）の規約

* `task_claim` が成功したタスクのみ、オーケストレーションに渡してよい。
* `ttl_seconds` を過ぎた claim は失効して再取得可能になる。
* `running` への遷移は claim 成功後に行う（または claim と同時に行う）。

---

## 5. ツールカーネル（Tool Kernel）

### 5.1 責務

ツールカーネルは「ツールを探し、説明し、実行する」基盤である。

* ツール定義の登録と検索（ToolRegistry）
* 入力検証、実行、出力検証（ToolExecutor）
* 実装参照（Implementation Reference）を介して実装を呼び出す
* DSLを導入する場合、DSLを読み込んでツール定義へ変換する（DSL Loader）

ツールカーネルは、タスクのキューやPDCAの概念を持たない。

### 5.2 ツール定義（Tool Definition）スキーマ

```yaml
ToolDefinition:
  tool_name: string
  description: string
  input_schema: object       # 入力の形式定義（JSON Schema等）
  output_schema: object      # 出力の形式定義（JSON Schema等）
  implementation_ref: string # 実装参照ID
  effects:                   # 副作用の宣言（説明必須）
    - string                 # 例: state.task_update, state.task_create, state.artifact_write, network.http
```

* `effects` は「このツールが起こしうる副作用の種類」を文字列で宣言する。
* 副作用がないツールは `effects: []` とする。

### 5.3 実装参照（Implementation Reference）

* `implementation_ref` はツール定義から生成された実行可能実装（関数）を指すための参照IDである。
* 例：DSLから `implementation_ref="impl:artifact_read_v1"` のような参照が生成される。
* ツールカーネルは `implementation_ref -> handler` のマッピングを保持する。

### 5.4 ツール実行コンテキスト（Tool Context）

ツール呼び出し時に渡されるコンテキストを定義する。

```yaml
ToolContext:
  runner_id: string
  task_id: string|null
  state_api: object          # 状態カーネルAPIへのアクセス（合成層が注入）
  now: string                # 実行時刻（ISO 8601）
```

* ツールが状態を更新する場合、`state_api` を通じて行う。
* これにより「副作用は状態カーネルに集約」できる。

### 5.5 ツールカーネル公開API（最小）

* `tool_invoke(tool_name, args, tool_context) -> result`
* `tool_describe(tool_name) -> ToolDefinition`
* `tool_search(query) -> tool_name[]`
* `tool_register(tool_definition, handler) -> void`

### 5.6 DSL の位置づけ（任意）

DSL は「ツールを自然言語で定義し、実装参照を生成して登録する」ための入力形式である。

* DSL を使わない場合：`tool_register` に直接 tool_definition と handler を渡す
* DSL を使う場合：DSL Loader が tool_definition と implementation_ref を生成し、tool_register に流す

---

## 6. オーケストレーションカーネル（Orchestration Kernel）

### 6.1 責務

オーケストレーションカーネルは、タスク処理の「やり方（実行戦略）」を提供する。

* LLM の呼び出し
* LLM からのツール呼び出し要求を受け取り、ツールカーネルへ委譲
* タスクを完了・失敗にするための進行管理
* 反復実行（PDCA等）の場合、Step と Step Context を管理

オーケストレーションカーネルは、永続化の実装を持たず、状態操作は状態カーネルAPI越しに行う。

### 6.2 Runner インターフェース

```yaml
Runner:
  run(task: Task, runner_context: RunnerContext) -> RunResult
```

#### 6.2.1 RunnerContext

```yaml
RunnerContext:
  runner_id: string
  state_api: object          # 状態カーネルAPI（合成層が注入）
  tool_api: object           # ツールカーネルAPI（合成層が注入）
  llm_api: object            # LLM呼び出しAPI（実装が注入）
```

#### 6.2.2 RunResult

```yaml
RunResult:
  user_output: string|null   # ユーザーへ返す文字列（main用途で使用）
  task_state: string         # done | failed | running
  artifact_refs:
    - string                 # 生成・参照したartifact_id
  error:
    code: string|null
    message: string|null
```

### 6.3 実行戦略（Orchestration Strategy）

本設計では最低限、以下2つの戦略を定義する。

1. **単発戦略（Single Turn Strategy）**

   * LLM を中心に1回でタスクを処理する
2. **PDCA戦略（PDCA Strategy）**

   * Step を複数回繰り返してタスク達成を目指す
   * 1 Step = 1 PDCA サイクル（Plan/Do/Check/Act）

戦略は Runner の実装として提供される。合成層はどのRunnerを使うか選ぶだけでよい。

---

## 7. PDCA戦略（PDCA Runner）の仕様

### 7.1 Step の定義

* 1 Step = 1 PDCA サイクル
* PDCA は Plan / Do / Check / Act の4フェーズ

### 7.2 Step Context（ステップ文脈）の仕様

反復実行のための持ち越しメモとして、Step Context を次の形式で固定する。

```yaml
StepContext:
  facts:                      # 確定した事実（短い箇条書き）
    - string
  open_issues:                # 未解決の論点（短い箇条書き）
    - string
  plan:                       # 次にやること（短い箇条書き）
    - string
  last_result: string         # 直近の結果要約（1段落）
  artifact_refs:              # 参照したartifact_id
    - string
  budget:
    remaining_steps: integer
    spent_steps: integer
```

* Step Context は **肥大化させない**（会話ログ丸ごと保持しない）
* 必要な詳細は ArtifactStore に保存して参照する

### 7.3 Budget（継続条件）

PDCA Runner は `remaining_steps > 0` の間だけ Step を実行する。

* `remaining_steps == 0` で未達の場合、**エラーとして終了**する（あなたの方針）

エラーコード例：

* `BUDGET_EXCEEDED`

### 7.4 PDCAフェーズ仕様

#### 7.4.1 Plan

入力：

* Task
* StepContext

出力：

* `step_goal: string`
* `step_success_criteria: string`

Plan は LLM により生成してよい。

#### 7.4.2 Do

目的：

* ツールの選択と実行
* 必要に応じて LLM による統合（要約・抽出・整形）

Do では LLM が **ツール呼び出し要求**を出し、それを Runner が受けてツールカーネルに委譲する。

ツール呼び出し主体：

* **LLM が tool call を指示する**
* Runner はその指示を実行する

#### 7.4.3 Check

目的：

* `step_success_criteria` に照らして達成判定
* 不足（gaps）を抽出

Check は LLM により実行してよい。

出力：

* `evaluation: string`（達成/未達の説明）
* `gaps: string[]`（不足の列挙）

#### 7.4.4 Act

目的：

* 次Stepに持ち越す Step Context を更新
* 必要なら状態カーネルに対する操作（タスク作成・更新、Artifact書き込み）を行う

状態カーネル操作は「ツール」として行う（あなたの方針）：

* `task_create` 相当
* `task_update` 相当
* `artifact_write` 相当

※ これらはツールカーネルに登録されたツールとして提供し、内部で state_api を呼ぶ。

---

## 8. notification タスク仕様

notification は「ユーザーへ通知するためのタスク種別」である。notification の payload を以下に固定する。

```yaml
NotificationPayload:
  message: string                  # ユーザーに見せたい本文（Markdown可）
  severity: string                 # info | warn | error
  related_task_id: string|null     # 関連する元タスク
  artifact_refs:
    - string                       # 追加資料
```

notification タスクの Task は以下のようになる。

```yaml
Task:
  task_type: notification
  payload: NotificationPayload
```

notification の処理は Runner（main用途）が行い、`payload.message` をユーザー出力として返す。

---

## 9. ツールセット（最小必須）

本設計で、最小構成として用意するツールは以下である。

### 9.1 状態操作ツール（状態カーネルAPIを呼ぶ薄いツール）

* `task.create`

  * effects: `state.task_create`
* `task.update`

  * effects: `state.task_update`
* `task.get`

  * effects: `[]`
* `task.list`

  * effects: `[]`
* `task.claim`

  * effects: `state.task_claim`（排他取得は状態更新であるため副作用扱い）
* `artifact.write`

  * effects: `state.artifact_write`
* `artifact.read`

  * effects: `[]`
* `artifact.search`

  * effects: `[]`

### 9.2 ドメインツール（用途に応じて追加）

* `extract`（抽出）
* `format_for_user`（ユーザー向け整形）

※ これらは副作用なしで実装するのが望ましいが必須ではない。

---

## 10. 合成層の構成例

### 10.1 シンプル合成（単発戦略、単一Runner）

* 状態カーネル
* ツールカーネル
* オーケストレーションカーネル（Single Turn Runner）
* 合成層（入力→task_create→task_claim→runner.run→出力）

用途：

* 最小のタスク駆動 + tool call を確認

### 10.2 PDCA合成（PDCA戦略）

* 状態カーネル
* ツールカーネル
* オーケストレーションカーネル（PDCA Runner）
* 合成層

用途：

* Step反復、Step Context、Budget を含む完全版

### 10.3 main / worker 合成（役割分離）

用語定義：

* **main**：ユーザー出力を返すRunnerの運用形態
* **worker**：ユーザー出力を返さず、必要なら notification タスクを作るRunnerの運用形態

合成としては「Runnerの運用ポリシー」を分けるだけで、カーネルの分割は変えない。

* main合成：`user_request` と `notification` を主に claim する
* worker合成：`work` を主に claim する

どちらも状態カーネルの `task_claim(filter)` の filter を変えるだけでよい。

---

## 11. タスク処理フロー（標準）

### 11.1 user_request のフロー（main）

1. クライアントが入力を送る
2. 合成層が `task_create(task_type="user_request", payload=...)`
3. 合成層が `task_claim(filter=user_request, claimer_id=main, ttl=...)`
4. 合成層が状態カーネルから Task を取得して Runner に渡す
5. Runner が LLM を呼び、必要なら tool call を行う（ツールカーネル経由）
6. Runner が `task_update` や `artifact_write` を必要に応じて実行する（ツールとして）
7. Runner が RunResult.user_output を返す
8. 合成層が `task_complete` または `task_fail` を反映
9. クライアントへ出力を返す

### 11.2 work のフロー（worker）

1. worker合成層が `task_claim(filter=work, claimer_id=worker_i, ttl=...)`
2. Runner が処理し、必要なら `task_create(task_type="notification", payload=...)` をツールとして実行
3. workerは user_output を返さない（null）
4. タスクを done または failed にする

### 11.3 notification のフロー（main）

1. main合成層が `task_claim(filter=notification, ...)`
2. Runner が payload.message を user_output として返す
3. タスクを done にする
4. クライアントへ表示する

---

## 12. エラーハンドリング規約

### 12.1 Budget枯渇（PDCA Runner）

* `remaining_steps == 0` で未達なら `task_fail` とし、RunResult.error を設定する

例：

```yaml
error:
  code: BUDGET_EXCEEDED
  message: "ステップ上限に達しました。"
```

### 12.2 ツールエラー

* tool_invoke が失敗した場合、Runner は失敗を Step Context に記録して次の行動を決めてよい
* 致命的な場合は `task_fail` とする

### 12.3 claim タイムアウト

* claim が失効したタスクは再取得されうる
* Runner は idempotent（同じ処理を繰り返しても破綻しにくい）に近づけることが望ましい
* 重要な副作用（artifact.write 等）は重複を許容するか、metadata で重複検知できる設計にする

---

## 13. セキュリティと安全性（最小）

* ツールカーネルは、ツール定義に `effects` を必須とし、LLM が呼ぶツールの副作用を可視化できるようにする
* オーケストレーションカーネルは、タスク種別ごとに「呼んでよいツール」を制限してよい（許可リスト方式）
* 状態カーネルAPIは、Runnerから直接呼ばず、ツールとして経由してもよい（監査しやすくなる）

---

## 14. 実装ガイド（最小順）

本設計は「核を別々に作って合成する」ことが目的なので、実装順も核単位で独立に進められる。

1. 状態カーネル：TaskStore / ArtifactStore / claim を完成
2. ツールカーネル：tool_register / tool_invoke を完成（state_api を注入できるようにする）
3. オーケストレーションカーネル：Single Turn Runner を完成（LLM + tool call）
4. PDCA Runner を追加（Step Context + Budget）
5. 合成層で main/worker を分ける（claim filter の違いだけで開始）

---

## 15. 固定仕様（この設計で変えない）

* 3カーネル（状態 / ツール / オーケストレーション）の分離
* カーネル間は公開APIのみで接続する
* PDCA戦略では「1 Step = 1 PDCAサイクル」
* Budget枯渇はエラーとして終了（現方針）

---

## 16. まとめ

本設計は「大きな核を育てる」のではなく、以下の **独立した3核**を先に成立させ、その合成で全体システムを構成する。

* 状態カーネル：状態の正、排他取得、永続化
* ツールカーネル：ツール定義・実行、入力/出力検証、DSL対応の受け皿
* オーケストレーションカーネル：LLMとツールを使った実行戦略（単発・PDCA）

これにより、PDCAやDSLやmain/workerのような要素は「核の肥大化」ではなく、**合成と戦略差し替え**として整理され、設計の矛盾や表記の揺れを最小に保ちながら拡張できる。
