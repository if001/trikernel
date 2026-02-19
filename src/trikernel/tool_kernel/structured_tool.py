from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .models import ToolDefinition, ToolContext


@dataclass
class StructuredTool:
    definition: ToolDefinition
    handler: Callable[..., Any]

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        *,
        tool_name: Optional[str] = None,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        effects: Optional[list[str]] = None,
    ) -> "StructuredTool":
        name = tool_name or fn.__name__
        definition = ToolDefinition(
            tool_name=name,
            description=description,
            input_schema=input_schema or {"type": "object", "properties": {}},
            output_schema=output_schema or {"type": "object", "properties": {}},
            effects=effects or [],
        )
        return cls(definition=definition, handler=fn)

    def invoke(self, args: Dict[str, Any], context: ToolContext) -> Any:
        signature = inspect.signature(self.handler)
        params = signature.parameters
        if "context" in params:
            return self.handler(**args, context=context)
        if "tool_context" in params:
            return self.handler(**args, tool_context=context)
        return self.handler(**args)
