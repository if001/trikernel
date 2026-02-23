from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from .models import ToolContext

_TOOL_CONTEXT: ContextVar[Optional[ToolContext]] = ContextVar(
    "trikernel_tool_context", default=None
)


def get_tool_context() -> Optional[ToolContext]:
    return _TOOL_CONTEXT.get()


def set_tool_context(context: Optional[ToolContext]) -> None:
    _TOOL_CONTEXT.set(context)


@contextmanager
def tool_context_scope(context: Optional[ToolContext]) -> Iterator[None]:
    token = _TOOL_CONTEXT.set(context)
    try:
        yield
    finally:
        _TOOL_CONTEXT.reset(token)
