## Added
- Allow `TrikernelSession` to disable background memory updates via `enable_memory_updates`.
- Add memory manage tools for profile/semantic/episodic/procedural stores.
- Add StateKernel Ollama LLM wrapper and config.
- Move LangMem memory manager into state_kernel and add MemoryKernel API.
## Fixed
- Run LangMem store updates on the store's owning event loop to avoid cross-loop futures.
- When LangGraph hits recursion limits, build a partial response from checkpointed tool results.
## Changed
- Inject memory context into tool discovery and agent prompts (profile/semantic/episodic).
- Enable embedding index config for memory store (Ollama embeddings).
- Tool memory API now routes through state_kernel helpers.
- MemoryKernel is resolved via StateKernelAPI; RunnerContext no longer carries it.

## Changed
- Centralize dotenv loading in `utils.env` to avoid scattered `.env` reads.
- Skip `.env` loading for message/memory store configs when `data_dir` is provided.
- Use file paths (not SQLAlchemy-style URLs) for AsyncSqlite store/checkpointer connections.

## Fixed
- Honor explicit data_dir for message/memory stores in tests and ensure memory store paths are created.

## Changed
- Align async flow tests with async-only message/memory store lifetimes.

## Changed
- Replace procedural memory store updates with prompt optimization via `create_prompt_optimizer`.

## Breaking Changes
- Removed DSL-based tool registration; tools are now defined via LangChain `StructuredTool`.
- Removed user profile artifact tools in favor of LangMem-backed memory tools.

## Added
- Background memory updates via LangMem store manager queued as worker tasks.
- LangMem-backed semantic/episodic/procedural memory tools with a persistent store.
- JSON-backed memory store with optional hybrid search indexing.
- LangGraph checkpointer-backed message history store.

## Changed
- Memory schemas moved into `state_kernel`.
- Session/worker constructors now require `store` and `tool_llm_api`; orchestration LLM uses explicit protocol types.
- Message store is async-only and requires `build_message_store()` for setup.
- LangMem store managers split semantic/profile/episodic/procedural namespaces with schema-specific configs.
- Message store uses `AsyncSqliteSaver` for checkpoints.
- Memory tools now provide search-only access; saves are handled by background memory updates.
- Tool kernel API now registers `BaseTool` instances directly.
- Tool schemas are defined with Pydantic args schemas.
- Removed protocol abstractions around LLM/Tool APIs to use concrete classes.
- Message store now only exposes a LangGraph checkpointer (no manual turn APIs).

## Fixed

## Removed
- Tool DSL loaders and YAML definitions.
- LLM/Tool protocol interfaces that hid LangChain/LangGraph dependencies.
- Legacy runners (SingleTurnRunner/ToolLoopRunner/PDCARunner) and their examples.
