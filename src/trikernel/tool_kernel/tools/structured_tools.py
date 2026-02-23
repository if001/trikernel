from __future__ import annotations

from typing import Callable, Optional, Type, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from ..context import get_tool_context


def build_structured_tool(
    func: Callable[..., object],
    *,
    name: str,
    description: str,
    args_schema: Optional[Type[BaseModel]] = None,
) -> StructuredTool:
    def _wrapped(**kwargs: Any) -> object:
        context = get_tool_context()
        if context is None:
            raise ValueError("tool_context is not set")
        return func(**kwargs, context=context)

    return StructuredTool.from_function(
        func=_wrapped,
        name=name,
        description=description,
        args_schema=args_schema,
    )
