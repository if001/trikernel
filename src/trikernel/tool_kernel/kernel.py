from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import ToolContext, ToolDefinition
from .structured_tool import StructuredTool


def _validate_input(schema: Dict[str, Any], args: Dict[str, Any]) -> None:
    required = schema.get("required", [])
    missing = [key for key in required if key not in args]
    if missing:
        raise ValueError(f"Missing required args: {missing}")


class ToolKernel:
    def __init__(self) -> None:
        self._tools: Dict[str, StructuredTool] = {}

    def tool_register(
        self,
        tool_definition: ToolDefinition,
        handler: Any,
    ) -> None:
        self._tools[tool_definition.tool_name] = StructuredTool(
            definition=tool_definition,
            handler=handler,
        )

    def tool_register_structured(self, tool: StructuredTool) -> None:
        self._tools[tool.definition.tool_name] = tool

    def tool_describe(self, tool_name: str) -> ToolDefinition:
        return self._tools[tool_name].definition

    def tool_search(self, query: str) -> List[str]:
        if not query:
            return list(self._tools.keys())
        query_lower = query.lower()
        return [
            name
            for name in self._tools.keys()
            if query_lower in name.lower()
            or query_lower in self._tools[name].definition.description.lower()
        ]

    def tool_invoke(
        self, tool_name: str, args: Dict[str, Any], tool_context: ToolContext
    ) -> Any:
        tool = self._tools[tool_name]
        _validate_input(tool.definition.input_schema, args)
        return tool.invoke(args, tool_context)

    def tool_list(self) -> List[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]
