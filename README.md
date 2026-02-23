# Trikernel

Trikernel is a three-kernel architecture for building LLM agents. It separates state, tools, and orchestration into independent kernels, then composes them with a thin layer.

## English

### Features

- Three-kernel separation: state, tool, orchestration.
- High-level `TrikernelSession` to hide task lifecycle details.
- Tools defined via DSL + Python functions.
- LLM via LangChain + Ollama.

### Install

```bash
uv pip install -e .
```

### Quickstart (Terminal)

```python
from ui.terminal import TerminalUI
from trikernel import TrikernelSession
from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.ollama import ToolOllamaLLM
from trikernel.tool_kernel.registry import register_default_tools
from trikernel.orchestration_kernel import OllamaLLM, ToolLoopRunner


def main() -> None:
    ui = TerminalUI()
    state = StateKernel()
    tool_kernel = ToolKernel()
    register_default_tools(tool_kernel)

    runner = ToolLoopRunner()
    llm = OllamaLLM()
    tool_llm = ToolOllamaLLM()
    session = TrikernelSession(state, tool_kernel, runner, llm, tool_llm)

    ui.write_output("Type a message. Empty input exits.")
    while True:
        for message in session.drain_notifications():
            ui.write_output(message)
        user_input = ui.read_input("message> ").strip()
        if not user_input:
            break
        result = session.send_message(user_input, stream=False)
        if result.stream_chunks:
            ui.write_stream(result.stream_chunks)
        else:
            ui.write_output(result.message or "")


if __name__ == "__main__":
    main()
```

### Adding Tools (DSL + Python)

1. Define the tool in a DSL file (YAML).
2. Implement the function in Python.
3. Register the tool via `build_tools_from_dsl`.

Note: Tool implementations should accept `context` as the last argument. It is injected by the kernel at runtime and provides access to state APIs and other execution context via `ToolContext`.

Core DSL files live under `src/trikernel/tool_kernel/dsl`.

### Task Payload Schemas

- `user_request`: `{"user_message": "..."}`
- `work`: `{"message": "...", "run_at": "...", "repeat_interval_seconds": 3600, "repeat_enabled": true}` (the worker/runner interprets this message; scheduling fields are optional)

### Work Tasks (Background)

Use `TrikernelSession.create_work_task()` to enqueue background tasks handled by workers.
Example payloads are user-defined; the worker/runner can use tools to fulfill them.

```python
task_id = session.create_work_task(
    {"message": "https://example.com を読んで要約して保存してください"},
    run_at="2025-01-01T12:00:00+00:00",
    repeat_every_seconds=3600,
    repeat_enabled=True,
)
```
Recurring work tasks enforce a minimum interval of 1 hour.

### Environment Variables

Use `.env` for configuration:

```
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_SMALL_MODEL=qwen3:4b
OLLAMA_EMBED_MODEL=nomic-embed-text
work_space_dir=/path/to/workspace
GOOGLE_API_KEY=your_api_key
GEMINI_MODEL=gemini-1.5-pro
TRIKERNEL_TIMEZONE=Asia/Tokyo
```

### Tests

```bash
python -m pytest
```

## 日本語

### 特徴

- 状態・ツール・オーケストレーションの三分割アーキテクチャ。
- `TrikernelSession` によりタスクの作成/claim/完了などを隠蔽。
- DSL + Python 関数でツールを追加可能。
- LLM は LangChain 経由で Ollama を使用。

### インストール

```bash
uv pip install -e .
```

### クイックスタート（ターミナル）

```python
from ui.terminal import TerminalUI
from trikernel import TrikernelSession
from trikernel.state_kernel.kernel import StateKernel
from trikernel.tool_kernel.kernel import ToolKernel
from trikernel.tool_kernel.ollama import ToolOllamaLLM
from trikernel.tool_kernel.registry import register_default_tools
from trikernel.orchestration_kernel import OllamaLLM, ToolLoopRunner


def main() -> None:
    ui = TerminalUI()
    state = StateKernel()
    tool_kernel = ToolKernel()
    register_default_tools(tool_kernel)

    runner = ToolLoopRunner()
    llm = OllamaLLM()
    tool_llm = ToolOllamaLLM()
    session = TrikernelSession(state, tool_kernel, runner, llm, tool_llm)

    ui.write_output("メッセージを入力してください。空入力で終了します。")
    while True:
        for message in session.drain_notifications():
            ui.write_output(message)
        user_input = ui.read_input("message> ").strip()
        if not user_input:
            break
        result = session.send_message(user_input, stream=False)
        if result.stream_chunks:
            ui.write_stream(result.stream_chunks)
        else:
            ui.write_output(result.message or "")


if __name__ == "__main__":
    main()
```

### ツール追加（DSL + Python）

1. DSL（YAML）でツールを定義。
2. Python で関数を実装。
3. `build_tools_from_dsl` で登録。

補足: ツール実装の関数は、最後の引数に `context` を追加してください。実行時にカーネルが注入し、`ToolContext` 経由で state API などにアクセスできます。

コアの DSL は `src/trikernel/tool_kernel/dsl` にあります。

### タスクの payload 形式

- `user_request`: `{"user_message": "..."}`
- `work`: `{"message": "...", "run_at": "...", "repeat_interval_seconds": 3600, "repeat_enabled": true}`（worker/runner がこのメッセージを解釈します。スケジュール系は任意）

### Work タスク（バックグラウンド）

`TrikernelSession.create_work_task()` で worker 用のタスクを投入できます。
payload の内容は任意で、worker/runner がツールを使って処理する想定です。

```python
task_id = session.create_work_task(
    {"message": "https://example.com を読んで要約して保存してください"},
    run_at="2025-01-01T12:00:00+00:00",
    repeat_every_seconds=3600,
    repeat_enabled=True,
)
```
定期実行の間隔は最小1時間です。

### 環境変数

`.env` で設定します。

```
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_SMALL_MODEL=qwen3:4b
OLLAMA_EMBED_MODEL=nomic-embed-text
work_space_dir=/path/to/workspace
GOOGLE_API_KEY=your_api_key
GEMINI_MODEL=gemini-1.5-pro
TRIKERNEL_TIMEZONE=Asia/Tokyo
```

### テスト

```bash
uv run python -m pytest
```
