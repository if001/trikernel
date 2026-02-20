from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .langchain_tools import build_structured_tool, tool_definition_from_structured
from .models import ToolContext, ToolDefinition
from .protocols import ToolAPI
from .structured_tool import TrikernelStructuredTool, adapt_langchain_tool


def _validate_input(schema: Dict[str, Any], args: Dict[str, Any]) -> None:
    required = schema.get("required", [])
    missing = [key for key in required if key not in args]
    if missing:
        raise ValueError(f"Missing required args: {missing}")


@dataclass
class ToolEntry:
    definition: ToolDefinition
    handler: Any
    structured_tool: TrikernelStructuredTool


class ToolKernel(ToolAPI):
    def __init__(self) -> None:
        self._tools: Dict[str, ToolEntry] = {}

    def tool_register(
        self,
        tool_definition: ToolDefinition,
        handler: Any,
    ) -> None:
        self._tools[tool_definition.tool_name] = ToolEntry(
            definition=tool_definition,
            handler=handler,
            structured_tool=build_structured_tool(tool_definition, handler),
        )

    def tool_register_structured(self, tool: TrikernelStructuredTool) -> None:
        structured_tool = tool
        if not hasattr(structured_tool, "as_langchain"):
            structured_tool = adapt_langchain_tool(structured_tool)  # type: ignore[arg-type]
        definition = tool_definition_from_structured(structured_tool)
        handler = _extract_handler(structured_tool)
        self._tools[definition.tool_name] = ToolEntry(
            definition=definition,
            handler=handler,
            structured_tool=structured_tool,
        )

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
        entry = self._tools[tool_name]
        _validate_input(entry.definition.input_schema, args)
        if entry.handler:
            return _invoke_handler(entry.handler, args, tool_context)
        return entry.structured_tool.invoke(args)

    def tool_list(self) -> List[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]

    def tool_descriptions(self) -> List[Dict[str, Any]]:
        return [
            {
                "tool_name": tool.tool_name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "effects": tool.effects,
            }
            for tool in self.tool_list()
        ]

    def tool_structured_list(self) -> List[TrikernelStructuredTool]:
        return [tool.structured_tool for tool in self._tools.values()]


def _extract_handler(tool: TrikernelStructuredTool) -> Optional[Any]:
    tool_impl = tool.as_langchain()
    handler = getattr(tool_impl, "func", None)
    if handler:
        return handler
    handler = getattr(tool_impl, "coroutine", None)
    if handler:
        return handler
    return getattr(tool_impl, "_run", None)


def _invoke_handler(handler: Any, args: Dict[str, Any], context: ToolContext) -> Any:
    signature = inspect.signature(handler)
    params = signature.parameters
    if "context" in params:
        return handler(**args, context=context)
    if "tool_context" in params:
        return handler(**args, tool_context=context)
    return handler(**args)
