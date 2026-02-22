# ADR: Trikernel Architecture Decisions

## Status
Active.

## Context
Trikernel is a library for running LLM-driven tasks with tools, background
workers, and artifact storage. Requirements include kernel separation, tool
DSL registration, and safe task execution with clear status transitions.

## Decisions

### ADR-001: Three-kernel separation (state/tool/orchestration)
**Decision**: Keep state, tool, and orchestration kernels independent and
composed by a thin execution layer.

**Rationale**: Clear boundaries enable swapping storage, tool engines, and
execution strategies without cross-kernel coupling.

**Consequences**:
- Kernels communicate only via public APIs.
- Composition/execution handles lifecycle and scheduling.

### ADR-002: High-level session API
**Decision**: Provide `TrikernelSession` as the primary user API, hiding task
and turn internals.

**Rationale**: Most users should not manage `Task` or `Turn` directly.

**Consequences**:
- Session wraps create/claim/run/finalize flows.
- Execution loop is internal to session.

### ADR-003: Scheduling fields live in payload
**Decision**: Store `run_at`, `repeat_interval_seconds`, and `repeat_enabled`
inside task payload rather than top-level Task fields.

**Rationale**: Scheduling is execution-level behavior and should not expand
the core state model.

**Consequences**:
- Execution reads schedule from payload.
- DSL schemas document scheduling fields.

### ADR-004: Centralized status transitions in execution
**Decision**: Execution layer owns state transitions and fallbacks
(timeouts, queued expiration, worker failures).

**Rationale**: Avoid scattered status logic across kernels and tools.

**Consequences**:
- Dispatcher and session manage transitions.
- Queued tasks expire (max 1 day).

### ADR-005: Tool input validation uses DSL schema
**Decision**: Validate tool inputs against JSON schema-like DSL, including
`required` and `oneOf/anyOf/allOf`.

**Rationale**: Provide consistent validation across tools.

**Consequences**:
- Validation runs on every tool invoke.
- Invalid args return errors (no tool execution).

### ADR-006: Tool context as last argument
**Decision**: Tools receive `context` as the last argument (keyword-only in
implementations).

**Rationale**: Keeps tool signatures consistent while allowing access to state
and execution context.

**Consequences**:
- Tool DSL does not expose `context`.
- Tool kernel injects `context` at call time.

### ADR-007: Logging and error handling policy
**Decision**: Log errors to file with rotation (10MB, 5 backups), console
logs for info/debug, and fail tasks on unrecoverable errors.

**Rationale**: Keep error visibility while minimizing user-facing leakage.

**Consequences**:
- Main/worker errors produce `failed` status.
- Log files are the source of detailed error info.

### ADR-008: No automatic retries by default
**Decision**: Do not retry tasks on failure/timeout; allow future policy changes.

**Rationale**: Simpler behavior and explicit failure handling.

**Consequences**:
- Failures are terminal unless user resubmits.
- Timeouts mark tasks failed.

## Open Questions
- Whether notifications should expire or be retained indefinitely.
- Whether to add structured logs to file output later.
