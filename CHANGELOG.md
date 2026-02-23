## [Unreleased]

### Breaking Changes
- Removed DSL-based tool registration; tools are now defined via LangChain `StructuredTool`.
- Removed user profile artifact tools in favor of LangMem-backed memory tools.

### Added
- LangMem-backed semantic/episodic/procedural memory tools with a persistent store.
- JSON-backed memory store with optional hybrid search indexing.
- LangGraph checkpointer-backed message history store.

### Changed
- Tool kernel API now registers `BaseTool` instances directly.
- Tool schemas are defined with Pydantic args schemas.
- Removed protocol abstractions around LLM/Tool APIs to use concrete classes.
- Message store now only exposes a LangGraph checkpointer (no manual turn APIs).

### Fixed

### Removed
- Tool DSL loaders and YAML definitions.
- LLM/Tool protocol interfaces that hid LangChain/LangGraph dependencies.
- Legacy runners (SingleTurnRunner/ToolLoopRunner/PDCARunner) and their examples.
