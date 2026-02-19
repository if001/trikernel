Protocol基底クラスを使い、実装は単純化し、あとで差し替えられるようにしてください。
3つのカーネルは独立しており、公開インターフェースを通してアクセスできます。
testを行えるように適切に分離、dryを意識し、実装を進めてください。

## ライブラリ
- langchain v1.0以降、langgraph v1.0以降を使うこと
- logger、richライブラリでログ出力すること
- .envで設定を保持すること
- llmはlangchain経由でollamaを利用すること



## 状態カーネル（State Kernel）
Task, Artifactはyaml or jsonファイルで行う。
(今後DBに拡張できるように実装を差し替えられるようにすること)

## ツールカーネル（Tool Kernel）
DSLから実装を作成する想定でしたが、ここでは実装はpythonの関数を作成し、利用すること
DSLは作成しますが、読み込んだDSLのうちtool_name,description,input_schema,output_schemaを使い、
StructuredTool.from_functionを使ってツールを作成しましょう

「5.3 実装参照（Implementation Reference）」はスキップしましょう。

## オーケストレーションカーネル（Orchestration Kernel）
単発戦略（Single Turn Strategy）を作成してください。
PDCA戦略（PDCA Strategy）は保留し、今後の拡張とします。

## 合成層
「11. タスク処理フロー（標準）」を参照してださい。

mainスレッドと別のスレッドとして、mainが起動し、mainがworkerを別スレッドで起動します。
スレッドはpythonのコルーチンを利用してください。
mainとworker間の通信はzmqを用います。
最大worker数は2として変更できるようにしてください。

以下UIについても実装を進めてください。
ユーザーの入力はターミナルから行うとします。また出力もターミナルに表示します。
これらは今後差し替える可能性があります。
同期出力とストリーム出力に対応してください。

それぞれ以下のディレクトリのは以下にtests, examplesディレクトリを配置
src/trikernel/orchestration_kernel
src/trikernel/state_kernel
src/trikernel/tool_kernel


docs/design.mdを参照し以下の実装を進めてください。
4.3 状態カーネル公開API（最小）
9.1 状態操作ツール（状態カーネルAPIを呼ぶ薄いツール）

オーケストレーションカーネル（Orchestration Kernel）の単発戦略（Single Turn Strategy）について、

OLLAMAの接続情報は以下を利用すること。
OLLAMA_BASE_URL=http://172.22.1.15:11434
OLLAMA_SMALL_MODEL=gemma3:4b
OLLAMA_MODEL=qwen3_8b_8192

以下は再度提示です。
- logger、richライブラリでログ出力すること
- .envで設定を保持すること

StructuredTool.from_functionでツールを作成していますが、
yaml定義のdslを作成し(tool_kernel以下にディレクトリを作成してください)
dslを読み込み、dslの定義tool_name,description,input_schema,output_schemaを使って、StructuredTool.from_functionでツールを作成するようにしてください。


compositionにmain.pyを作成し、作成したこれまでの流れを実行できるようにしてください。
UIは以下です。
ユーザーの入力はターミナルから行うとします。また出力もターミナルに表示します。
これらは今後差し替える可能性があります。
同期出力とストリーム出力に対応してください。

orchestration_kernelのui.pyは不要のはずです。






src/trikernel/composition/main.pyでは、
docs/design.mdの「11. タスク処理フロー（標準）」が実現できているか
実装を確認してください。

アーティファクトやタスクの保存ディレクトリは.envで管理してください。
tool_kernel/dslはcoreのツールとして、
別ファイルにユーザー定義のdslを作成します。


web検索用のqueryの作成ツール、queryによりweb検索を行い一覧とsnippetを取得、urlでページの内容を取得
SIMPLE_CLIENT_BASE_URL=http://172.22.1.15:8000
web検索用のqueryでは、ユーザー入力と履歴、OLLAMA_SMALL_MODELを使って作成してください。

エンドポイントは以下です。
- /list queryによりweb検索を行い一覧とsnippetを取得
requestはListRequest
responseはListResponse

- /page urlでページの内容を取得
requestはPageRequest
responseはPageResponse

class ListRequest(BaseModel):
    q: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)


class ListResponse(BaseModel):
    query: str
    k: int
    results: List[SearchResult]

class SearchResult:
    rank: int
    title: str
    url: str
    snippet: Optional[str] = None
    published_date: Optional[str] = None

class PageRequest(BaseModel):
    urls: str = Field(..., min_length=1)

class DocOut(BaseModel):
    url: str
    title: str
    markdown: str

class PageResponse(BaseModel):
    docs: List[DocOut]






PDCA戦略について、以下の修正をしてください。
Do：Discoverで得た Step Toolset のみ実行可能
ユーザー定義のtoolも実行できるようにしましょう。




ユーザーとアシスタントとの会話履歴を保持する機能を追加しましょう。

- State Kernel（状態カーネル）：履歴の永続化と「最新5ターン取得API」
- Tool Kernel（ツールカーネル）：履歴参照を“ツール”として提供
- Orchestration Kernel（オーケストレーションカーネル）：
  - main は「応答生成の入力として履歴を常に注入」する

State Kernel の公開API（履歴用）
追加する最小APIは以下
- turn_append_user(conversation_id, user_message, related_task_id) -> turn_id
- turn_set_assistant(turn_id, assistant_message, artifacts, metadata) -> Turn
- turn_list_recent(conversation_id, limit) -> Turn[]（limit=5で使用）
turn_append_userとturn_set_assistantを分けていますがまとめても良いかもしれません
tool_search や PDCA で使うのは基本 turn_list_recent だけ
mainのときのみ履歴を利用し、workerは履歴を利用する必要はありません。

応答は常に直近5ターンを使うとして、
PDCA戦略のplanとdiscoverでturn_list_recent("default", 5) を取り、LLM入力に注入しましょう。
メッセージの保存は適宜おこなってください。
Step Context に履歴そのものを入れないとします


以下Anyではなく型かinterfaceを使うようにしましょう。
@dataclass
class RunnerContext:
    runner_id: str
    state_api: Any
    tool_api: Any
    llm_api: Any
    stream: bool = False

src/trikernel/tool_kernel/kernel.pyを修正しましょう。
class ToolKernel:
    def tool_search(self, query: str) -> List[str]:

toolの検索は、

TYPE_CHECKINGで解決しました。

では、次に
src/trikernel/orchestration_kernel/ollama.py
について修正しましょう。
urllibでリクエストしていますが、
"langchain-ollama>=1.0.1"を使うように修正しましょう。

1. PDCAの各ステップのLLMのpromptを改善してください。
何を行うべきか明確にしましょう。
LLM呼び出しは独立で行われています。したがって、PDCAのDoを行ってくださいのような指示は、
このシステム全体の文脈がわからなければ意味のわからないものです。このような指示は避け、独立したLLMで行うべきことを明確にしてください。

2. state_tools.yamlのタスクに関するdescriptionと引数の説明が分かりづらいので修正しましょう。
タスクについて、特にtask_type:workは、mainのPDCAではなく、別スレッドでworkerしバックグラウンドで実行するものです。すなわち、処理に時間がかかるとされるものを実行する場合に使います。また、listやcomplieteはtask:workをすでに実行しているときに、処理が終わったかや待ちとなっているタスクがないかを確認するための用途に使います。

3. artifactに関しても、何をするものかいつ使うものかなど、具体的にしてください。

_do_stepを修正しましょう。一般的なtoolを利用するagentに寄せましょう。
以下のようなステップとなっているはずです。間違っていれば適宜対応してください。
1. llmにprompt+tool_listを入力し、出力としてユーザーへの応答とtoolを選択したかが出力されます。
2. toolが選択されていれば、tool実行. promtp+toolの結果を入力し、ユーザーへの応答を得ます。
3. toolが選択されていなければ、ユーザーへの応答を返す。






_act_stepを修正しましょう。
_act_stepを単純化し最低限行うようにします。
step_contextの更新のみ、tool callは行わない。
PDCARunner.runがworkerによる実行場合は、ユーザー通知タスクを作成する（最終stepのみ、毎サイクル呼ぶと通知がうるさい）。
その他、設計やコンセプトからずれる、対応漏れがあるなどの懸念があれば提案してください。


 _check_stepの呼び出しのtool_resultsは削除しています。
 _check_stepでtool_resultsは必要か検討してください。

StepContextは1stepにおけるstateです。
未使用のものがないか、過剰なものはないか、足りていないものはないか確認してください。
各フィールドの存在する目的を考慮し検討してください。



do_response.user_outputが空/短い場合、すなわち最終出力が空ということです。
_check_stepは、最終出力の確認であり、空の場合や出力に十分でない場合は再度stepを行うべきです。
toolの呼び出し結果が失敗した場合は、最終出力にツールの呼び出しが失敗した内容が含まれる想定です。
toolの呼び出しの失敗理由が、呼び出し方にあるのであれば呼び出し方を修正し再度stepを行い、呼ばれた側のエラーであれば
どうすることもできないので、その旨を最終出力に含めるしかありません。

「1stepにおけるstate」は訂正します。「runコンテキスト」が正しいです。


tool呼び出しに失敗した場合、失敗理由は「呼び出し側のミス(引数を間違えているなど)」or「呼ばれる側のエラー」の2つを想定して、
失敗の理由を返しているか確認してください。
失敗している場合、失敗理由をもとに再度ツールを選び、引数を渡す必要があります。そのような設計となっているか確認してください。
runにおけるコンテキスト「StepContext」に含まれる必要があるか、1step内で完結するか検討してください。

artifact.write結果を拾うのは、結果を拾うと判断し、ツールとしてartifact.readを選択した場合です。
artifact.searchで取り出し、artifact.readを行う想定です。このような実装になっているか確認してください。


1. Tool失敗理由の扱い
提案の方向性で修正を進めてください。
