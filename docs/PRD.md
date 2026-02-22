# PRD: Trikernel (3-kernel LLM agent runtime)

## Summary
Trikernel is a Python library that provides a 3-kernel LLM agent runtime
(state/tool/orchestration) with a high-level session API. It supports tool
definition via DSL + Python, task execution with main/worker separation, and
artifact storage/search.

## Goals
- Provide a stable library API that hides internal task/turn details.
- Keep kernels independent and swappable.
- Allow users to add tools via DSL + Python without touching core kernels.
- Support background work tasks and notifications.
- Provide safe scheduling (run_at) and recurring work (min 1 hour).
- Centralize status transitions and fallback handling in execution.
- Provide error logging with log rotation.

## Non-goals
- UI design or UX for chat clients.
- Network protocols or external service hosting.
- Full sandboxing or security hardening.
- Automatic retries/backoff (default is no retry).
- Monitoring/metrics collection.

## Users and Use Cases
### Primary Users
- Python developers integrating LLM workflows in CLI/Discord bots.
- Users who want to register custom tools without editing core code.

### Example Use Cases
- CLI chat: user_request -> main runner -> response.
- Discord read channel: create work task -> worker runs -> notification to channel.
- Web page summarization: fetch page -> store artifact -> extract summary.

## Product Requirements

### Functional Requirements
1. **Kernels**
   - State kernel persists tasks, artifacts, turns and provides claim/transition API.
   - Tool kernel registers DSL-defined tools and validates inputs.
   - Orchestration kernel runs tasks with LLM + tools.

2. **High-level API**
   - `TrikernelSession.send_message()` for user_request tasks.
   - `TrikernelSession.create_work_task()` for work tasks.
   - `TrikernelSession.start_workers()` for background processing.

3. **Tooling**
   - DSL (YAML/JSON) defines tool name/description/schemas.
   - Python implementations receive `context` as last argument.
   - Tool input validation must respect `required` and `oneOf/anyOf/allOf`.

4. **Scheduling**
   - `run_at` is stored in task payload.
   - `run_at` must be future and within 1 year; otherwise reject at tool/session.
   - Recurring work tasks use `repeat_interval_seconds` + `repeat_enabled`.
   - Minimum repeat interval is 1 hour.

5. **Status Transitions**
   - `queued -> running -> done/failed`.
   - Execution layer handles timeouts and cleanup.
   - Queued tasks expire after a shared timeout (max 1 day).
   - Worker timeout and queue timeout set tasks to failed.
   - user_request claim failure sets task to failed.

6. **Artifacts**
   - Artifacts saved as separate files and searchable by hybrid search.
   - `artifact.list` returns id, metadata, created_at, body preview.
   - `artifact.extract` uses LLM to extract requested info.

7. **Logging**
   - Errors are logged to file with rotation (10MB, 5 backups).
   - Console logs are allowed (info/debug).

### Non-Functional Requirements
- Use Python 4-space indentation, standard naming, explicit type hints for public APIs.
- Avoid `Any` in public kernel interfaces where possible.
- Compatible with langchain >= 1.0 and langgraph >= 1.0.
- Use Ollama via LangChain.
- Configuration via `.env`.

## Constraints
- State/Tool/Orchestration kernels must remain independent.
- Tool definitions should be created via `StructuredTool.from_function`.
- Tools for file access must be constrained to `work_space_dir`.

## Success Criteria
- Users can use `TrikernelSession` without knowing task/turn internals.
- Tools can be registered via DSL and validated (including oneOf schemas).
- work tasks can be scheduled and (optionally) repeated safely.
- Execution layer recovers by failing stale queued/running tasks.
- Errors are visible in logs with rotation enabled.

## Out of Scope / Future
- Retry/backoff strategies.
- Job cancellation and restart policies.
- Metrics and monitoring pipeline.
- Multi-tenant isolation or full sandboxing.
